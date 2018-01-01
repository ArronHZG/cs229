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
import threading
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
para = np.array(
    [1.3214190747119177, -0.20304530218574768, 0.05693978958304286, 0.025090380718581117, 1.3194338829139964,
     1.1377088319284556, 4.743479574979944, 0.015085797177806013, -0.8501607216661536, 0.30843859004075524,
     -0.0122030391340392, -0.3492701566578825, 0.023990672175949905, -0.588074211424225])
# para=np.ones(numFeatures)
# 步长
step = 0.00000001
errorJ = 10000
accept = 4000


def run(step, errorJ, accept):
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
                para[j] = para[j] + step * np.dot(err(trainData.target, para, trainData.features),
                                                  trainData.features[:, j])


for i in range(64):
    t = threading.Thread(target=run, args=(0.00000001, 10000, 0.0000000001))
    t.start()
print(para.tolist())
mse = mean_squared_error(testData.target, h(para, testData.features))
evs = explained_variance_score(testData.target, h(para, testData.features))
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))
