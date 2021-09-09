# --coding--:utf-8 --
from PyQt5.QtCore import *

from PyQt5.QtWidgets import *

from PyQt5.QtGui import *

import cv2

from Ui_main import Ui_MainWindow

class MainWindow(QMainWindow, Ui_MainWindow):

def __init__(self, parent=None):

super(MainWindow, self).__init__(parent)

self.setupUi(self)

self.timer_camera = QTimer(self)

self.cap = cv2.VideoCapture(0)

self.timer_camera.timeout.connect(self.show_pic)

self.timer_camera.start(10)

def show_pic(self):

success, frame=self.cap.read()

if success:

show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)

self.label.setPixmap(QPixmap.fromImage(showImage))

self.timer_camera.start(10)

if __name__=='__main__':

import sys

app=QApplication(sys.argv)

window=MainWindow()

window.show()

sys.exit(app.exec_())