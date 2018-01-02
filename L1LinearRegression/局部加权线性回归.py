# -*- coding: UTF-8 -*-
'''
@author: Arron
@license: (C) Copyright 2018-2025, Node Supply Chain Manager Corporation Limited.
@contact: hou.zg@foxmail.com
@software: import
@file: 局部加权线性回归.py
@time: 2018/1/2 0002 19:59
'''
import numpy as np
import math
import L1LinearRegression.dataSource as pd
from sklearn.metrics import mean_squared_error, explained_variance_score

data = pd.Data(pd.features1, pd.target1)
numData = len(data.target)
numTraining = numData
# 设置x0=1
data.features = np.concatenate([np.ones((1, numData)), data.features.transpose()]).transpose()
numFeatures = len(data.features[0])



def h(para, x):
    return np.dot(para, x.transpose())


def err(y, para, x):
    return y - h(para, x)


def J(para, features, target):
    return np.sum(0.5 * (h(para, features) - target) ** 2)


def wi(xi, x, tao):
    return math.e ** (-(xi - x) ** 2 / (2 * tao ** 2))


def Jw(para, features, target, tao, x):
    w = wi(features, x, tao)
    return np.sum(np.dot(w.transpose(),(h(para, features) - target) ** 2))


def run(step, errorJ, accept,x):
    k = 0
    while errorJ[0]-errorJ[1] >= accept:
        for i in range(numTraining):
            for j in range(numFeatures):
                errorJ[0],errorJ[1] = errorJ[1],Jw(para, data.features, data.target,1,x)
                para[j] = para[j] + step * np.dot(err(data.target, para, data.features),
                                                  data.features[:, j])
        print('errorJ', errorJ)

para = np.array([-100,0.1,100])
x=np.array([1,3000,4])
step = 0.000000001
errorJ = [10000,9000]
accept = 1
run(step, errorJ, accept, x)
print(para.tolist())
print(np.sum(np.dot(para,x.transpose())))
