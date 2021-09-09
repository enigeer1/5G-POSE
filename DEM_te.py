# --coding--:utf-8 --
import cv2
img = cv2.imread('xin1.jpg', cv2.IMREAD_COLOR)
# print(img)
# cv2.imshow('image', img)
# K = cv2.waitKey(0)
with open('xin1.jpg', 'rb') as file:
    f = file.readline()
    print(type(f))

    s = str(f, encoding='unicode')
    print(s)