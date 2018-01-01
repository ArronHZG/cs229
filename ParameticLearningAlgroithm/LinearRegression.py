# -*- coding: UTF-8 -*-
'''
@author: Arron
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: hou.zg@foxmail.com
@software: import
@file: LinearRegression.py
@time: 2017/12/30 0030 18:28

'''
import numpy as np
import ParameticLearningAlgroithm.dataSource as pd
from sklearn.metrics import mean_squared_error, explained_variance_score
import matplotlib.pyplot as plt

# pd.features=pd.features[:2]
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


# 线性拟合参数
para = np.array([1.1265787082229892, -0.17700206366312762, 0.06251803626578507, 0.07120968676534453, 1.145487214461865,
                 1.071263866659266, 2.8441501643306384, 0.05593989309585012, -0.40110051778467004, 0.2605999597885871,
                 -0.010046513153710013, -0.2872173734016564, 0.043761582366230865, -0.7580278959019118])
# para=np.ones(numFeatures)
# 步长
step = 0.00000001

errorJ = 10000
accept = 4000
k = 0
while errorJ >= accept:
    k += 1
    if k == 100:
        print('errorJ', errorJ)
        print(para.tolist())
        k = 0
    for i in range(numTraining):
        # print('errorJ', errorJ)
        for j in range(numFeatures):
            # print('(i,j)', i, j)
            errorJ = J(para, trainData.features, trainData.target)
            para[j] = para[j] + step * np.dot(err(trainData.target, para, trainData.features), trainData.features[:, j])

print(para.tolist())
mse = mean_squared_error(testData.target, h(para, testData.features))
evs = explained_variance_score(testData.target, h(para, testData.features))
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))
