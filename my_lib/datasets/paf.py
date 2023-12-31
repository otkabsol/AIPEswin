# coding=utf-8
"""Implement Part Affinity Fields
:param centerA: int with shape (2,), centerA will pointed by centerB.
:param centerB: int with shape (2,), centerB will point to centerA.
:param accumulate_vec_map: one channel of paf.
:param count: store how many pafs overlaped in one coordinate of accumulate_vec_map.
:param params_transform: store the value of stride and crop_szie_y, crop_size_x                 
"""
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc, ndimage


def putVecMaps(centerA, centerB, accumulate_vec_map, count, grid_y, grid_x, stride):
    centerA = centerA.astype(float)
    centerB = centerB.astype(float)

    thre = 1  # limb width
    centerB = centerB / stride #映射到特征图中
    centerA = centerA / stride

    limb_vec = centerB - centerA
    norm = np.linalg.norm(limb_vec)#求范数
    if (norm == 0.0):
        # print 'limb is too short, ignore it...'
        return accumulate_vec_map, count
    limb_vec_unit = limb_vec / norm #单位向量
    # print 'limb unit vector: {}'.format(limb_vec_unit)

    # To make sure not beyond the border of this two points
    min_x = max(int(round(min(centerA[0], centerB[0]) - thre)), 0)# 得到所有可能区域
    max_x = min(int(round(max(centerA[0], centerB[0]) + thre)), grid_x)
    min_y = max(int(round(min(centerA[1], centerB[1]) - thre)), 0)
    max_y = min(int(round(max(centerA[1], centerB[1]) + thre)), grid_y)

    range_x = list(range(int(min_x), int(max_x), 1))
    range_y = list(range(int(min_y), int(max_y), 1))
    xx, yy = np.meshgrid(range_x, range_y)
    ba_x = xx - centerA[0]  # the vector from (x,y) to centerA 根据位置判断是否在该区域上（分别得到X和Y方向的）
    ba_y = yy - centerA[1]
    limb_width = np.abs(ba_x * limb_vec_unit[1] - ba_y * limb_vec_unit[0]) #向量叉乘根据阈值选择赋值区域，任何向量与单位向量的叉乘即为四边形的面积
    mask = limb_width < thre  # mask is 2D # 小于阈值的表示在该区域上

    vec_map = np.copy(accumulate_vec_map) * 0.0 #本次计算

    vec_map[yy, xx] = np.repeat(mask[:, :, np.newaxis], 2, axis=2)
    vec_map[yy, xx] *= limb_vec_unit[np.newaxis, np.newaxis, :] #在该区域上的都用对应的方向向量表示（根据mask结果表示是否在）

    mask = np.logical_or.reduce(
        (np.abs(vec_map[:, :, 0]) > 0, np.abs(vec_map[:, :, 1]) > 0)) #在特征图中（46*46）中 哪些区域是该躯干所在区域

    accumulate_vec_map = np.multiply(
        accumulate_vec_map, count[:, :, np.newaxis]) #每次返回的accumulate_vec_map都是平均值，现在还原成实际值
    accumulate_vec_map += vec_map # 加上当前关键点位置形成的向量
    count[mask == True] += 1 # 该区域计算次数都+1

    mask = count == 0

    count[mask == True] = 1 # 没有被计算过的地方就等于自身（因为一会要除法）

    accumulate_vec_map = np.divide(accumulate_vec_map, count[:, :, np.newaxis])# 算平均向量
    count[mask == True] = 0 # 还原回去

    return accumulate_vec_map, count
