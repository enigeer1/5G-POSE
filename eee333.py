# --coding--:utf-8 --
import math

#
# def angle_of_vector(v1, v2):
#     pi = 3.1415
#     vector_prod = v1[0] * v2[0] + v1[1] * v2[1]
#     length_prod = math.sqrt(pow(v1[0], 2) + pow(v1[1], 2)) * math.sqrt(pow(v2[0], 2) + pow(v2[1], 2))
#     cos = vector_prod * 1.0 / (length_prod * 1.0 + 1e-6)
#     print((math.acos(cos) / pi) * 180)
#     return (math.acos(cos) / pi) * 180
#
#
# v1 = (4, 6)
# v2 = (6, 4)
# angle_of_vector(v1, v2)
# import numpy as np
# import math
# a = np.array([1, 0, 0])
# b = np.array([0, 1, 1])
# cos_ab = a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))
# print(np.arccos(cos_ab)*180/np.pi)
#
# a = (1, 2)
# print(a[1])
import numpy as np


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
    print('夹角=%f°' % angle_d)


a = (2, 3, 8)
b = (2, 3, 4)
cal_angel(a, b)