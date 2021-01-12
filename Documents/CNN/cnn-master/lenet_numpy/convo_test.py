# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 12:47:32 2021

@author: ASUS
"""
import os
from convolution import *
from relu import *
from sigmoid import *
from fc import *
from maxpool import *
from softmax import *
from Data import Data
import timeit

mainPath = os.path.dirname(os.path.abspath(__file__)) #file path main.py
workPath = os.path.split(mainPath) #path working folder (whole file project)
imagePath = "data_kain"

layer_time = []
data = Data(workPath, imagePath)
trainSet, trainLabel, testSet, testLabel = data.load()
"""
resolusi 48 x 48
kernel ukuran 5 x 5 menghasilkan feature map 24 x 24
kernel ukuran 3 x 3 menghasilkan feature map 25 x 25
input node = luas dari resolusi
layer node = luas dari resolusi layer X jumlah layer
"""
conv1 = CONV_LAYER((6, 48, 48), (6, 1, 5, 5), (2304, 13824), pad=2, stride=1,) #filename=b_dir+"conv0.npz")
relu1 = RELU_LAYER()
# Sub-sampling-1
pool1 = MAX_POOL_LAYER(stride=2)

conv2 = CONV_LAYER((6, 24, 24), (6, 1, 3, 3), (576, 3456), pad=2, stride=1,) #filename=b_dir+"conv0.npz")
relu2 = RELU_LAYER()
# Sub-sampling-1
pool2 = MAX_POOL_LAYER(stride=2)

"""
pool 2 out put size (240, 6, 13, 13)

"""

start = timeit.default_timer()
conv1.forward(trainSet)
relu1.forward(conv1.feature_map)
pool1.forward(relu1.feature_map)

conv2.forward(pool1.feature_map)
relu2.forward(conv2.feature_map)
pool2.forward(relu2.feature_map)

stop = timeit.default_timer()
layer_time += [stop-start]

#print(conv1.feature_map)
#print(relu1.feature_map)  
print("durasi 2 layer ekstrasi ",layer_time)