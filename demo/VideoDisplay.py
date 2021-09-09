# --coding--:utf-8 --
import cv2
import threading
from PyQt5.QtCore import QFile
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
import numpy as np


class Display:
    def __init__(self, ui, mainWnd):
        self.ui = ui
        self.mainWnd = mainWnd

        # 默认视频源为相机
        self.ui.radioButtonCam.setChecked(True)
        self.isCamera = True

        # 信号槽设置
        ui.Open.clicked.connect(self.Open)
        ui.Close.clicked.connect(self.Close)
        ui.radioButtonCam.clicked.connect(self.radioButtonCam)
        ui.radioButtonFile.clicked.connect(self.radioButtonFile)
        # 增加TextEdit控件

        # 增加角度显示功能
        ui.pushButton.clicked.connect(self.show_right_eblow_angel)
        ui.pushButton_2.clicked.connect(self.show_left_eblow_angel)
        ui.pushButton_3.clicked.connect(self.show_right_armpit_angel)
        ui.pushButton_4.clicked.connect(self.show_left_armpit_angel)
        # 表情显示按钮        # 增加表情
        ui.pushButton_5.clicked.connect(self.Open1)

        # 创建一个关闭事件并设为未触发
        self.stopEvent = threading.Event()
        self.stopEvent.clear()

    def show_right_eblow_angel(self):
        with open('show_right_eblow_angel.csv', 'r', encoding="utf-8") as f:
            msg = f.read()
            self.ui.textEdit.setPlainText(msg)

    def show_left_eblow_angel(self):
        with open('show_left_eblow_angel.csv', 'r', encoding="utf-8") as f:
            msg = f.read()
            self.ui.textEdit_2.setPlainText(msg)

    def show_right_armpit_angel(self):
        with open('list_angle_right_armpit.csv', 'r', encoding="utf-8") as f:
            msg = f.read()
            self.ui.textEdit_3.setPlainText(msg)

    def show_left_armpit_angel(self):
        with open('list_angle_left_armpit.csv', 'r', encoding="utf-8") as f:
            msg = f.read()
            self.ui.textEdit_4.setPlainText(msg)

    def radioButtonCam(self):
        self.isCamera = True

    def radioButtonFile(self):
        self.isCamera = False

    def Open(self):
        if not self.isCamera:
            self.fileName, self.fileType = QFileDialog.getOpenFileName(self.mainWnd, 'Choose file', '', '*.mp4')
            self.cap = cv2.VideoCapture(self.fileName)
            self.frameRate = self.cap.get(cv2.CAP_PROP_FPS)
        else:
            # 下面两种rtsp格式都是支持的
            # cap = cv2.VideoCapture("rtsp://admin:Supcon1304@172.20.1.126/main/Channels/1")
            # self.cap = cv2.VideoCapture("rtsp://admin:Supcon1304@172.20.1.126:554/h264/ch1/main/av_stream")
            # self.cap = cv2.VideoCapture("rtmp://220.196.249.183:7592/camera1")
            self.cap = cv2.VideoCapture("rtmp://220.196.249.183:7592/camera1")

        # 创建视频显示线程
        th = threading.Thread(target=self.Display)
        th.start()

    def Open1(self):
        self.cap = cv2.VideoCapture("rtmp://220.196.249.183:7592/camera2")
        # 创建视频显示线程
        th = threading.Thread(target=self.Display1)
        th.start()

    def Close(self):
        # 关闭事件设为触发，关闭视频播放
        self.stopEvent.set()

    def Display(self):
        self.ui.Open.setEnabled(False)
        self.ui.Close.setEnabled(True)

        while self.cap.isOpened():
            success, frame = self.cap.read()
            # RGB转BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            print(np.array(frame).shape)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            self.ui.DispalyLabel.setPixmap(QPixmap.fromImage(img))

            if self.isCamera:
                cv2.waitKey(1)
            else:
                cv2.waitKey(int(1000 / self.frameRate))

            # 判断关闭事件是否已触发
            if True == self.stopEvent.is_set():
                # 关闭事件置为未触发，清空显示label
                self.stopEvent.clear()
                self.ui.DispalyLabel.clear()
                self.ui.Close.setEnabled(False)
                self.ui.Open.setEnabled(True)
                break

    def Display1(self):
        self.ui.pushButton_5.setEnabled(False)

        while self.cap.isOpened():
            success, frame = self.cap.read()
            # RGB转BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            print(np.array(frame).shape)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            self.ui.DispalyLabel.setPixmap(QPixmap.fromImage(img))

            if self.isCamera:
                cv2.waitKey(1)
            else:
                cv2.waitKey(int(1000 / self.frameRate))

            # 判断关闭事件是否已触发
            if True == self.stopEvent.is_set():
                # 关闭事件置为未触发，清空显示label
                self.stopEvent.clear()
                self.ui.DispalyLabel.clear()
                self.ui.pushButton_5.setEnabled(True)
                break