import socket
import struct
import numpy
import cv2
import queue
import time
import threading

# 接包 套接字
s_recv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s_recv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
# 绑定地址和端口
s_recv.bind(('172.23.63.0', 7550))
# 开启监听
s_recv.listen(128)
# client 为客户端对象
# addr 为客户端地址
ss, addr = s_recv.accept()

# 发包套接字
client_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
addr_server = ('192.168.8.152', 7551)
client_send.connect(addr_server)


def receive_data():
    while True:
        data = ss.recv(20480)
        if not data:
            break
        # 图片的长度  图片的宽高
        client_send.send(data)
        # cv2.imshow('窗口', image)
        # cv2.waitKey(1)


receive_data()





