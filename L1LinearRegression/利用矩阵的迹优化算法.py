# -*- coding: UTF-8 -*-
'''
@author: Arron
@license: (C) Copyright 2018-2025, Node Supply Chain Manager Corporation Limited.
@contact: hou.zg@foxmail.com
@software: import
@file: 利用矩阵的迹优化算法.py
@time: 2018/1/2 0002 13:59
'''
# θ = (XT*X)^−1 * XT * y
import numpy as np
import L1LinearRegression.dataSource as pd
from sklearn.metrics import mean_squared_error, explained_variance_score
import threading
import matplotlib.pyplot as plt

# pd.features=pd.features[:2]
data = pd.Data(pd.features1, pd.target1)
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

def theta(x,y):
    x=np.mat(x)
    y=np.mat(y)
    return (x.T * x).I * x.T * y.T

para=theta(data.features,data.target)
para=para.T.getA()
para=para[0]
print(para.tolist())
# print(target)
# for i in range(len(target)):
#     print(target[i],data.target[i])

mse = mean_squared_error(testData.target, h(para, testData.features))
evs = explained_variance_score(testData.target, h(para, testData.features))
j=J(para,data.features,data.target)
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))
print("J() = ", round(j, 2))

'''
[-70.43460183227586, 0.06384337561663134, 103.4360465116271]
Mean squared error = 26.63
Explained variance score = 1.0
J() =  722.07


[36.491103280361116, -0.10717055656035364, 0.0463952195297986, 0.02086023953217151, 2.6885613993179356, -17.795758660308916, 3.8047524602581624, 0.0007510617033193782, -1.4757587965198014, 0.30565503833908636, -0.012329346305268524, -0.9534635546905188, 0.009392512722189415, -0.5254666329007901]
Mean squared error = 14.83
Explained variance score = 0.46
J() =  5540.14
'''