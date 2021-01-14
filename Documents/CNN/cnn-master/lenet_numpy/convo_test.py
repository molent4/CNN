# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 12:47:32 2021

@author: ASUS
"""
import os
from convolution import *
from relu import *
from sigmoid import *
from fullyconnect import *
from maxpool import *
from softmax import *
from Activation_Softmax import *
from Data import Data
from loss import *
import timeit

mainPath = os.path.dirname(os.path.abspath(__file__)) #file path main.py
workPath = os.path.split(mainPath) #path working folder (whole file project)
imagePath = "data_kain"

layer_time = []
data = Data(workPath, imagePath)
trainSet, trainLabel, testSet, testLabel = data.load()

Train_target = np.zeros((len(trainLabel),data.jum_kelas))
Train_target[np.arange(len(trainLabel)), trainLabel[range(0, len(trainLabel))]] = 1
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

fc3 = FC_LAYER(120, (1014, 120), )#filename=b_dir+"fc6.npz")
# Fully Connected-2
fc4 = FC_LAYER(84, (120, 84), )#filename=b_dir+"fc8.npz")
# Fully Connected-3
fc5 = FC_LAYER(4, (84, 4), )#filename=b_dir+"fc10.npz")
sf = ACTIVATION_SOFTMAX()
loss = Loss_CategoricalCrossentropy()
sigmoid1 = SIGMOID_LAYER()

relu3 = RELU_LAYER()


"""

pool 1 output size (240, 6, 24, 24)
pool 2 output size (240, 6, 13, 13)

"""

start = timeit.default_timer()
conv1.forward(trainSet)
relu1.forward(conv1.feature_map)
pool1.forward(relu1.output)

conv2.forward(pool1.feature_map)
relu2.forward(conv2.feature_map)
pool2.forward(relu2.output)

temp = pool2.feature_map

fc3.forward(temp.reshape(pool2.feature_map.shape[0], pool2.feature_map.shape[1]*pool2.feature_map.shape[2]*pool2.feature_map.shape[3]))
relu3.forward(fc3.output)

fc4.forward(relu3.output)
sigmoid1.forward(fc4.output)

fc5.forward(sigmoid1.output)

sf.forward(fc5.output)
loss.calculate(sf.output, Train_target)

stop = timeit.default_timer()
layer_time += [stop-start]

#print(conv1.feature_map)
#print(relu1.feature_map)  
print("durasi 2 layer ekstrasi ",layer_time)

loss.backward(sf.output, Train_target)
sf.backward(loss.dinputs)
fc5.backward(sf.dinputs)
sigmoid1.backward(fc5.delta_X)
fc4.backward(sigmoid1.delta_X)

print(sf.dinputs)