#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse
import subprocess as sp
import csv
import cv2 as cv
import numpy as np
import mediapipe as mp
from utils import CvFpsCalc
import socket
import math
import time
import os



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
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
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
#############dddddddddd
    use_brect = args.use_brect

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)


    # rtmpUrl = "rtmp://139.196.208.10:7592/camera2"
    rtmpUrl = "rtmp://127.0.0.1:1935/live"
    # rtmpUrl = "rtmp://139.196.208.10:7503/live/livestream"
    # rtmpUrl = "rtmp://139.196.208.10:8080/camera1"

    camera_path = 0
    cap = cv.VideoCapture(camera_path)

    # Get video information`
    fps = int(cap.get(cv.CAP_PROP_FPS))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    command = [
        # 'F:\\Downloads\\Compressed\\ffmpeg-4.4-full_build\\bin\\ffmpeg',
        'D:\\ffmpeg_build\\bin\\ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', "{}x{}".format(width, height),
        '-r', '25',
        # '-r', str(fps),
        '-i', '-',
        '-max_delay', '1',
        '-g', '0',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast',
            # '-segment_times', '5',
            # '-bufsize', '64M',
            # '-maxrate', "4M",
        # 'zerolatency', '1',
        # '-fflags', 'nobuffer',
        # '-b', '900000',
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
        # if results.pose_landmarks is not None:
        #     # print("results.pose_landmarks", results.pose_landmarks)
        #     # 外接矩形の計算
        #     brect = calc_bounding_rect(debug_image, results.pose_landmarks)
        #     # 描画
        #     debug_image = draw_landmarks(debug_image, results.pose_landmarks,
        #                                  upper_body_only)
        #     debug_image = draw_bounding_rect(use_brect, debug_image, brect)
        #
        # cv.putText(debug_image, "FPS:" + str(display_fps), (10, 30),
        #            cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)

        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # 画面反映 #############################################################
        cv.imshow('MediaPipe Pose Demo', debug_image)
        p.stdin.write(debug_image.tobytes())
        # print(debug_image)
    cap.release()
    cv.destroyAllWindows()


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]




def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # 外接矩形
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 255, 0), 2)

    return image


if __name__ == '__main__':
    main()
