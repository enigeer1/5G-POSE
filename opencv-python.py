# 导入cv模块
import cv2 as cv
# 读取图像
img = cv.imread('33.png')
# cv.imshow('read.jpg', img)
# # 转化成灰度图像
# gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
# # 显示灰度图像
# cv.imshow("gray_img", gray_img)

def face_detect_demo():
    gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    cv.imshow('result1:', gray)

    face_detect = cv.CascadeClassifier('D:/opencvD/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml')
    face = face_detect.detectMultiScale(gray, 1.1)
    for x,y,w,h in face:
        cv.rectangle(img,(x,y), (x+w, y+h), color=(0,255,0), thickness=2)
    cv.imshow('result:', img)


face_detect_demo()


cv.waitKey(0)
cv.destroyAllWindows()


