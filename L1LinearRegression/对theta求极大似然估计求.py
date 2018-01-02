# -*- coding: UTF-8 -*-
'''
@author: Arron
@license: (C) Copyright 2018-2025, Node Supply Chain Manager Corporation Limited.
@contact: hou.zg@foxmail.com
@software: import
@file: 对theta求极大似然估计求.py
@time: 2018/1/2 0002 16:10
'''

import numpy as np
import L1LinearRegression.dataSource as pd
from sklearn.metrics import mean_squared_error, explained_variance_score
data = pd.Data(pd.features, pd.target)
numData = len(data.target)
numTraining = int(0.8 * numData)
# 设置x0=1
data.features = np.concatenate([np.ones((1, numData)), data.features.transpose()]).transpose()
numFeatures = len(data.features[0])
trainData = pd.Data(data.features[:numTraining], data.target[:numTraining])
testData = pd.Data(data.features[numTraining:], data.target[numTraining:])


def h(para, x):
    return np.dot(para, x.transpose())


def err(y, para, x):
    return y - h(para, x)


def J(para, features, target):
    return np.sum(0.5 * (h(para, features) - target) ** 2)

#计算极大似然估计后,求最大值
# 最终目标求J 可以使用梯度下降或矩阵迹运算
