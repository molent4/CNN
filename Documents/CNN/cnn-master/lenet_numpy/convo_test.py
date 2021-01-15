# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 12:47:32 2021

@author: ASUS
"""
import os
from convolution import *
from relu import *
from sigmoid import *
from fullyconnects import *
from maxpool import *
from softmax import *
from Activation_Softmax import *
from Data import Data
from loss import *
import timeit
from Optimizer import *
from tqdm import tqdm



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
"""
Convo Input:
(layer_size, kernel_size, fan, **params)
    layer_size: tuple consisting (depth, height, width)
    kernel_size: tuple consisting (number_of_kernels, inp_depth, inp_height, inp_width)
    fan: tuple of number of nodes in previous layer and this layer
    params: directory consists of pad_len and stride,
    filename (to load weights from file)
"""

conv1 = CONV_LAYER((6, 48, 48), (6, 3, 5, 5), (2304, 13824), pad=2, stride=1,) #filename=b_dir+"conv0.npz")
relu1 = RELU_LAYER()
# Sub-sampling-1
pool1 = MAX_POOL_LAYER(stride=2)

conv2 = CONV_LAYER((6, 24, 24), (6, 6, 3, 3), (576, 3456), pad=2, stride=1,) #filename=b_dir+"conv0.npz")
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
#optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-7)
#optimizer = Optimizer_RMSprop(learning_rate=0.02, decay=1e-5, rho=0.999)
optimizer = Optimizer_SGD(decay=8e-8, momentum=0.9)
#optimizer = Optimizer_Adagrad(decay=1e-4)

"""

pool 1 output size (240, 6, 24, 24)
pool 2 output size (240, 6, 13, 13)

"""
epochs = 1001
error_array = []
for epoch in tqdm(range(epochs)):
    
    start = timeit.default_timer()
    conv1.forward(trainSet)
    relu1.forward(conv1.feature_map)
    pool1.forward(relu1.output)
    
    conv2.forward(pool1.feature_map)
    relu2.forward(conv2.feature_map)
    pool2.forward(relu2.output)
    
    #print("conv 1",conv1.feature_map.shape)
    #print("pool 1",pool1.feature_map.shape)
    
    #print("conv 2",conv2.feature_map.shape)
    #print("pool 2",pool2.feature_map.shape)
    
    temp = pool2.feature_map
    
    fc3.forward(temp.reshape(pool2.feature_map.shape[0], pool2.feature_map.shape[1]*pool2.feature_map.shape[2]*pool2.feature_map.shape[3]))
    relu3.forward(fc3.output)
    
    fc4.forward(relu3.output)
    sigmoid1.forward(fc4.output)
    
    fc5.forward(sigmoid1.output)
    
    sf.forward(fc5.output)
    error = loss.calculate(sf.output, trainLabel)
    error_array.append(error)
    

    loss.backward(sf.output, trainLabel)
    sf.backward(loss.dinputs)
    fc5.backward(sf.dinputs)
    sigmoid1.backward(fc5.delta_X)
    fc4.backward(sigmoid1.delta_X)
    relu3.backward(fc4.delta_X)
    fc3.backward(relu3.delta_X)
    

    pool2.backward(fc3.delta_X)
    relu2.backward(pool2.delta_X)
    conv2.backward(relu2.delta_X)
    
    
    pool1.backward(conv2.delta_X)
    relu1.backward(pool1.delta_X)
    conv1.backward(relu1.delta_X)
    
    
    
    optimizer.pre_update_params()

    optimizer.update_params(fc5)
    optimizer.update_params(fc4)
    optimizer.update_params(fc3)
    optimizer.update_params(conv1)
    optimizer.update_params(conv2)

    
    stop = timeit.default_timer()
    optimizer.post_update_params()
    layer_time += [stop-start]
    predictions = np.argmax(sf.output, axis=1)
    if len(trainLabel.shape) == 2:
        trainLabel = np.argmax(trainLable, axis=1)
        # np.argmax return indexs of max value each row (axis 1)
        # np.argmax return array of index refering to position of maximun value along axis 1
    accuracy = np.mean(predictions==trainLabel)
    
    if not epoch % 2:
        print('\n'+ f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {error:.3f}, ' +
              f'lr: {optimizer.current_learning_rate}')

    

    
