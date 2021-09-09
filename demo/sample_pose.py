#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse
import subprocess as sp

import cv2 as cv
import numpy as np
import mediapipe as mp
from utils import CvFpsCalc
import socket
import math


def cal_angel(x, y):
    x = np.array(x)
    y = np.array(y)
    l_x = np.sqrt(x.dot(x))
    l_y = np.sqrt(y.dot(y))
    print('向量的模=', l_x, l_y)

    # 计算两个向量的点积
    dian = x.dot(y)
    print('向量的点积=', dian)

    # 计算夹角的cos值：
    cos_ = dian / (l_x * l_y)
    print('夹角的cos值=', cos_)

    # 求得夹角（弧度制）：
    angle_hu = np.arccos(cos_)
    print('夹角（弧度制）=', angle_hu)

    # 转换为角度值：
    angle_d = angle_hu * 180 / np.pi
    with open('./angle.txt', "w", newline="") as f:
        f.write(str(angle_d))
        f.write('\n')
    print('夹角=%f°' % angle_d)
    return angle_d


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--upper_body_only', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.9)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.9)

    parser.add_argument('--use_brect', action='store_true')

    args = parser.parse_args()

    return args


def main():
    # 引数解析 #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    upper_body_only = args.upper_body_only
    # print("upper_body_only::::", upper_body_only)
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = args.use_brect

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)


    rtmpUrl = "rtmp://120.55.171.119:7592/camera1"
    camera_path = 0
    cap = cv.VideoCapture(camera_path)

    # Get video information`
    fps = int(cap.get(cv.CAP_PROP_FPS))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    command = [
        # 'F:\\Downloads\\Compressed\\ffmpeg-4.4-full_build\\bin\\ffmpeg',
        'G:\\ffmpeg_build\\bin\\ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', "{}x{}".format(width, height),
        '-r', "15",
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast',
        '-f', 'flv',
        # rtspUrl]
        rtmpUrl]


    # 管道配置
    p = sp.Popen(command, stdin=sp.PIPE)
    # モデルロード #############################################################
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        upper_body_only=upper_body_only,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    while True:
        display_fps = cvFpsCalc.get()
        # カメラキャプチャ #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # ミラー表示
        debug_image = copy.deepcopy(image)

        # 検出実施 #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = pose.process(image)

        # 描画 ################################################################
        if results.pose_landmarks is not None:
            # print("results.pose_landmarks", results.pose_landmarks)
            # 外接矩形の計算
            brect = calc_bounding_rect(debug_image, results.pose_landmarks)
            # 描画
            debug_image = draw_landmarks(debug_image, results.pose_landmarks,
                                         upper_body_only)
            debug_image = draw_bounding_rect(use_brect, debug_image, brect)
            
        cv.putText(debug_image, "FPS:" + str(display_fps), (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)

        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # 画面反映 #############################################################
        cv.imshow('MediaPipe Pose Demo', debug_image)
        p.stdin.write(debug_image.tobytes())
    cap.release()
    cv.destroyAllWindows()


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    # print("landmark_array::::", landmark_array)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def draw_landmarks(image, landmarks, upper_body_only, visibility_th=0.5):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    #################################################################################################################
    # print("landmarks:::"+"\n", landmarks)
    for index, landmark in enumerate(landmarks.landmark):
        # print("index, landmark:::::", index, landmark)
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        # print("landmark_x::", landmark_x)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # print("landmark_y::", landmark_y)
        landmark_z = landmark.z
        # print("landmark_z::", landmark_z)
        landmark_point.append([landmark.visibility, (landmark_x, landmark_y)])
        # print(landmark_point.__sizeof__())

        if landmark.visibility < visibility_th:
            continue

        if index == 0:  # 鼻
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 1:  # 右目：目頭
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 2:  # 右目：瞳
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 3:  # 右目：目尻
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 4:  # 左目：目頭
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 5:  # 左目：瞳
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 6:  # 左目：目尻
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 7:  # 右耳
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 8:  # 左耳
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 9:  # 口：左端
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 10:  # 口：左端
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 11:  # 右肩
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 12:  # 左肩
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 13:  # 右肘
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 14:  # 左肘
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 15:  # 右手首
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 16:  # 左手首
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 17:  # 右手1(外側端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 18:  # 左手1(外側端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 19:  # 右手2(先端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 20:  # 左手2(先端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 21:  # 右手3(内側端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 22:  # 左手3(内側端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 23:  # 腰(右側)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 24:  # 腰(左側)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 25:  # 右ひざ 膝盖
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 26:  # 左ひざ
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 27:  # 右足首
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 28:  # 左足首
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 29:  # 右かかと 脚后跟
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 30:  # 左かかと
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 31:  # 右つま先   脚趾
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 32:  # 左つま先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        # print("upper_body_only::::::", upper_body_only)
        if not upper_body_only:
            # print("landmark_z,::::", landmark_z)
            cv.putText(image, "z:" + str(round(landmark_z, 3)),
                       (landmark_x - 10, landmark_y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                       cv.LINE_AA)
    list_angle = []
    if len(landmark_point) > 0:
        print("landmark_point::", landmark_point)
        # 右目
        if landmark_point[1][0] > visibility_th and landmark_point[2][
                0] > visibility_th:
            cv.line(image, landmark_point[1][1], landmark_point[2][1],
                    (0, 255, 0), 2)
        if landmark_point[2][0] > visibility_th and landmark_point[3][
                0] > visibility_th:
            cv.line(image, landmark_point[2][1], landmark_point[3][1],
                    (0, 255, 0), 2)

        # 左目
        if landmark_point[4][0] > visibility_th and landmark_point[5][
                0] > visibility_th:
            cv.line(image, landmark_point[4][1], landmark_point[5][1],
                    (0, 255, 0), 2)
        if landmark_point[5][0] > visibility_th and landmark_point[6][
                0] > visibility_th:
            cv.line(image, landmark_point[5][1], landmark_point[6][1],
                    (0, 255, 0), 2)

        # 口
        if landmark_point[9][0] > visibility_th and landmark_point[10][
                0] > visibility_th:
            cv.line(image, landmark_point[9][1], landmark_point[10][1],
                    (0, 255, 0), 2)

        # 肩
        if landmark_point[11][0] > visibility_th and landmark_point[12][
                0] > visibility_th:
            cv.line(image, landmark_point[11][1], landmark_point[12][1],
                    (0, 255, 0), 2)
            print("右肩的landmark_z", landmark_z)

        # 右腕
        if landmark_point[11][0] > visibility_th and landmark_point[13][0] > visibility_th:
            # 右肩=================================================================
            cv.line(image, landmark_point[11][1], landmark_point[13][1],
                    (255, 0, 0), 2)
            # 添加右上臂向量
            print("右肘的landmark_z", landmark_z)
            right_shoulder = np.array([landmark_point[11][1][0], landmark_point[11][1][1], int(landmark_z)])
            right_elbow_point = np.array([landmark_point[13][1][0], landmark_point[13][1][1], int(landmark_z)])
            right_up_elbow_vector = right_shoulder - right_elbow_point
            list_angle.append(right_up_elbow_vector)

        if landmark_point[13][0] > visibility_th and landmark_point[15][
                0] > visibility_th:
            cv.line(image, landmark_point[13][1], landmark_point[15][1],
                    (255, 0, 0), 2)
            print("右腕的landmark_z", landmark_z)
            # 添加右下臂向量
            right_elbow_point = np.array([landmark_point[13][1][0], landmark_point[13][1][1], int(landmark_z)])
            right_hand = np.array([landmark_point[15][1][0], landmark_point[15][1][1], int(landmark_z)])
            right_down_elbow_vector = right_hand - right_elbow_point
            list_angle.append(right_down_elbow_vector)
            print("right_down_elbow_vector::", right_down_elbow_vector)
        # 左腕
        if landmark_point[12][0] > visibility_th and landmark_point[14][
                0] > visibility_th:
            cv.line(image, landmark_point[12][1], landmark_point[14][1],
                    (0, 255, 0), 2)
            # 左上臂向量
            left_shoulder = np.array([landmark_point[12][1][0], landmark_point[12][1][1], int(landmark_z)])
            left_elbow_point = np.array([landmark_point[14][1][0], landmark_point[14][1][1], int(landmark_z)])
            left_up_elbow_vector = left_shoulder - left_elbow_point
            list_angle.append(left_up_elbow_vector)

        if landmark_point[14][0] > visibility_th and landmark_point[16][
                0] > visibility_th:
            cv.line(image, landmark_point[14][1], landmark_point[16][1],
                    (0, 255, 0), 2)
            # 左下臂向量
            left_elbow_point = np.array([landmark_point[14][1][0], landmark_point[14][1][1], int(landmark_z)])
            left_hand = np.array([landmark_point[16][1][0], landmark_point[16][1][1], int(landmark_z)])
            left_down_elbow_vector = left_hand - left_elbow_point
            list_angle.append(left_down_elbow_vector)

        # 右手
        if landmark_point[15][0] > visibility_th and landmark_point[17][
                0] > visibility_th:
            cv.line(image, landmark_point[15][1], landmark_point[17][1],
                    (0, 255, 0), 2)
        if landmark_point[17][0] > visibility_th and landmark_point[19][
                0] > visibility_th:
            cv.line(image, landmark_point[17][1], landmark_point[19][1],
                    (0, 255, 0), 2)
        if landmark_point[19][0] > visibility_th and landmark_point[21][
                0] > visibility_th:
            cv.line(image, landmark_point[19][1], landmark_point[21][1],
                    (0, 255, 0), 2)
        if landmark_point[21][0] > visibility_th and landmark_point[15][
                0] > visibility_th:
            cv.line(image, landmark_point[21][1], landmark_point[15][1],
                    (0, 255, 0), 2)

        # 左手
        if landmark_point[16][0] > visibility_th and landmark_point[18][
                0] > visibility_th:
            cv.line(image, landmark_point[16][1], landmark_point[18][1],
                    (0, 25, 0), 2)
        if landmark_point[18][0] > visibility_th and landmark_point[20][
                0] > visibility_th:
            cv.line(image, landmark_point[18][1], landmark_point[20][1],
                    (0, 255, 0), 2)
        if landmark_point[20][0] > visibility_th and landmark_point[22][
                0] > visibility_th:
            cv.line(image, landmark_point[20][1], landmark_point[22][1],
                    (0, 255, 0), 2)
        if landmark_point[22][0] > visibility_th and landmark_point[16][
                0] > visibility_th:
            cv.line(image, landmark_point[22][1], landmark_point[16][1],
                    (0, 255, 0), 2)

        # 胴体
        if landmark_point[11][0] > visibility_th and landmark_point[23][
                0] > visibility_th:
            # 右肩到腰
            # right_shoulder = np.array([landmark_point[11][1][0], landmark_point[11][1][1], int(landmark_z)])
            # right_waist = np.array([landmark_point[23][1][0], landmark_point[23][1][1], int(landmark_z)])
            # right_shoulder_to_waist_vector = right_shoulder - right_waist
            # list_angle.append(right_shoulder_to_waist_vector)
            # 右上臂向量
            # right_shoulder = np.array([landmark_point[11][1][0], landmark_point[11][1][1], int(landmark_z)])
            # right_elbow_point = np.array([landmark_point[13][1][0], landmark_point[13][1][1], int(landmark_z)])
            # right_up_elbow_vector = right_shoulder - right_elbow_point
            # list_angle.append(right_up_elbow_vector)
            cv.line(image, landmark_point[11][1], landmark_point[23][1], (0, 255, 0), 2)
        if landmark_point[12][0] > visibility_th and landmark_point[24][
                0] > visibility_th:
            cv.line(image, landmark_point[12][1], landmark_point[24][1],
                    (0, 255, 0), 2)
        if landmark_point[23][0] > visibility_th and landmark_point[24][
                0] > visibility_th:
            cv.line(image, landmark_point[23][1], landmark_point[24][1],
                    (0, 255, 0), 2)
            # 左肩到腰
            # left_shoulder = np.array([landmark_point[12][1][0], landmark_point[12][1][1], int(landmark_z)])
            # left_waist = np.array([landmark_point[24][1][0], landmark_point[24][1][1], int(landmark_z)])
            # left_shoulder_to_waist_vector = left_shoulder - left_waist
            # list_angle.append(left_shoulder_to_waist_vector)
            # 左上臂向量
            # left_shoulder = np.array([landmark_point[12][1][0], landmark_point[12][1][1], int(landmark_z)])
            # left_elbow_point = np.array([landmark_point[14][1][0], landmark_point[14][1][1], int(landmark_z)])
            # left_up_elbow_vector = left_shoulder - left_elbow_point
            # list_angle.append(left_up_elbow_vector)


        if len(landmark_point) > 25:
            # 右足
            if landmark_point[23][0] > visibility_th and landmark_point[25][
                    0] > visibility_th:
                cv.line(image, landmark_point[23][1], landmark_point[25][1],
                        (0, 255, 0), 2)
            if landmark_point[25][0] > visibility_th and landmark_point[27][
                    0] > visibility_th:
                cv.line(image, landmark_point[25][1], landmark_point[27][1],
                        (0, 255, 0), 2)
            # 增加右大腿向量

            # right_waist = np.array([landmark_point[23][1][0], landmark_point[23][1][1], int(landmark_z)])
            # right_knee_point = np.array([landmark_point[25][1][0], landmark_point[25][1][1], int(landmark_z)])
            # right_knee_up_vector = right_waist - right_knee_point
            # list_angle.append(right_knee_up_vector)

            if landmark_point[27][0] > visibility_th and landmark_point[29][
                    0] > visibility_th:
                cv.line(image, landmark_point[27][1], landmark_point[29][1],
                        (0, 255, 0), 2)
            # 增加右小腿向量

            # right_knee_point = np.array([landmark_point[25][1][0], landmark_point[27][1][1], int(landmark_z)])
            # right_heel = np.array([landmark_point[25][1][0], landmark_point[25][1][1], int(landmark_z)])
            # right_knee_down_vector = right_heel - right_knee_point
            # list_angle.append(right_knee_down_vector)
            if landmark_point[29][0] > visibility_th and landmark_point[31][
                    0] > visibility_th:
                cv.line(image, landmark_point[29][1], landmark_point[31][1],
                        (0, 255, 0), 2)

            # 左足
            if landmark_point[24][0] > visibility_th and landmark_point[26][
                    0] > visibility_th:
                cv.line(image, landmark_point[24][1], landmark_point[26][1],
                        (0, 255, 0), 2)
            if landmark_point[26][0] > visibility_th and landmark_point[28][
                    0] > visibility_th:
                cv.line(image, landmark_point[26][1], landmark_point[28][1],
                        (0, 255, 0), 2)
            if landmark_point[28][0] > visibility_th and landmark_point[30][
                    0] > visibility_th:
                cv.line(image, landmark_point[28][1], landmark_point[30][1],
                        (0, 255, 0), 2)
            # 添加左大腿向量
            # left_waist = np.array([landmark_point[24][1][0], landmark_point[24][1][1], int(landmark_z)])
            # left_knee_point = np.array([landmark_point[26][1][0], landmark_point[26][1][1], int(landmark_z)])
            # left_knee_up_vector = left_waist - left_knee_point
            # list_angle.append(left_knee_up_vector)
            # # 添加左小腿向量
            # left_knee_point = np.array([landmark_point[26][1][0], landmark_point[26][1][1], int(landmark_z)])
            # left_heel = np.array([landmark_point[28][1][0], landmark_point[28][1][1], int(landmark_z)])
            # left_knee_down_vector = right_heel - right_knee_point
            # list_angle.append(left_knee_down_vector)
            if landmark_point[30][0] > visibility_th and landmark_point[32][
                    0] > visibility_th:
                cv.line(image, landmark_point[30][1], landmark_point[32][1],
                        (0, 255, 0), 2)
    print("")
    # 右臂0 1
    # right_eblow_angle = cal_angel(list_angle[0], list_angle[1])
    # cv.putText(image, "right_eblow_angle::" + str(right_eblow_angle), (5, 50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    # 左臂2 3
    # left_eblow_angel = cal_angel(list_angle[2], list_angle[3])
    # cv.putText(image, "left_eblow_angel::" + str(left_eblow_angel), (5, 70), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    # 右胳肢窝 4 5
    # right_armpit_angel = cal_angel(list_angle[4], list_angle[5])
    # cv.putText(image, "right_armpit_angel::" + str(right_armpit_angel), (5, 90), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    # # 左胳肢窝 6 7
    # right_armpit_angel = cal_angel(list_angle[6], list_angle[7])
    # cv.putText(image, "right_armpit_angel::" + str(right_armpit_angel), (5, 110), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    # # 右腿8 9
    # right_knee_angel = cal_angel(list_angle[8], list_angle[9])
    # cv.putText(image, "right_knee_angel::" + str(right_knee_angel), (5, 130), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    # # 左腿10 11
    # leg_knee_angel = cal_angel(list_angle[10], list_angle[11])
    # cv.putText(image, "leg_knee_angel::" + str(leg_knee_angel), (5, 150), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    # print("list_angle:::::::", np.array(list_angle).shape)
    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # 外接矩形
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 255, 0), 2)

    return image


if __name__ == '__main__':
    main()
