import collections
from math import *

training_data = [[170, 57, 32, 0],
                 [192, 95, 28, 1],
                 [150, 45, 30, 0],
                 [170, 65, 29, 1],
                 [175, 78, 35, 1],
                 [185, 90, 32, 1],
                 [170, 65, 28, 0],
                 [155, 48, 31, 0],
                 [160, 55, 30, 0],
                 [182, 80, 30, 1],
                 [175, 69, 28, 0],
                 [180, 80, 27, 1],
                 [160, 50, 31, 0],
                 [175, 72, 30, 1]]


class Node:
    def __init__(self):
        pass


def cal_distance(vec1, vec2):
    tmp = 0
    for i in range(len(vec1)):
        tmp += (vec1[i] - vec2[i]) ** 2
    return sqrt(tmp)
