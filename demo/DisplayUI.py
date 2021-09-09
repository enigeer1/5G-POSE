# --coding--:utf-8 --
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(853, 734)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.radioButtonCam = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButtonCam.setGeometry(QtCore.QRect(140, 540, 121, 31))
        self.radioButtonCam.setObjectName("radioButtonCam")

        self.radioButtonFile = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButtonFile.setGeometry(QtCore.QRect(140, 580, 121, 31))
        self.radioButtonFile.setObjectName("radioButtonFile")

        self.Open = QtWidgets.QPushButton(self.centralwidget)
        self.Open.setGeometry(QtCore.QRect(350, 560, 121, 41))
        self.Open.setObjectName("Open")

        self.Close = QtWidgets.QPushButton(self.centralwidget)
        self.Close.setGeometry(QtCore.QRect(550, 560, 111, 41))
        self.Close.setObjectName("Close")

        self.DispalyLabel = QtWidgets.QLabel(self.centralwidget)
        # self.DispalyLabel.setGeometry(QtCore.QRect(71, 44, 711, 411))
        self.DispalyLabel.setGeometry(QtCore.QRect(71, 44, 711, 411))
        self.DispalyLabel.setMouseTracking(False)
        self.DispalyLabel.setText("")
        self.DispalyLabel.setObjectName("DispalyLabel")
        # 添加计算右肘的按钮
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(840, 390, 141, 23))
        self.pushButton.setObjectName("pushButton")

        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(830, 440, 151, 171))
        self.textEdit.setObjectName("textEdit")
        # 添加计算左肘的按钮
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(1020, 390, 151, 23))
        self.pushButton_2.setObjectName("pushButton_2")

        self.textEdit_2 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_2.setGeometry(QtCore.QRect(1020, 440, 151, 171))
        self.textEdit_2.setObjectName("textEdit_2")
        # 添加右腋
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(830, 170, 151, 23))
        self.pushButton_3.setObjectName("pushButton_3")

        self.textEdit_3 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_3.setGeometry(QtCore.QRect(830, 210, 151, 161))
        self.textEdit_3.setObjectName("textEdit_3")
        # 添加左腋
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(1020, 170, 151, 23))
        self.pushButton_4.setObjectName("pushButton_4")

        self.textEdit_4 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_4.setGeometry(QtCore.QRect(1020, 210, 151, 161))
        self.textEdit_4.setObjectName("textEdit_4")
        # 添加表情识别控件按按钮
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(830, 80, 75, 23))
        self.pushButton_5.setObjectName("pushButton_5")


        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 853, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.radioButtonCam.setText(_translate("MainWindow", "camera"))
        self.radioButtonFile.setText(_translate("MainWindow", "local file"))
        self.Open.setText(_translate("MainWindow", "Open"))
        self.Close.setText(_translate("MainWindow", "Close"))
        # 添加计算右肘的按钮
        self.pushButton.setText(_translate("MainWindow", "show_right_eblow_angel"))
        self.pushButton_2.setText(_translate("MainWindow", "show_left_eblow_angel"))
        self.pushButton_3.setText(_translate("MainWindow", "show_right_Armpit_angel "))
        self.pushButton_4.setText(_translate("MainWindow", "show_left_Armpit_angel "))
        # 添加表情识别控件
        self.pushButton_5.setText(_translate("MainWindow", "show_face"))
