from mnist import MNIST
import os, sys
import numpy as np
from CNNlenet5 import *
import timeit
from Data import Data



if __name__ == '__main__':
    mainPath = os.path.dirname(os.path.abspath(__file__)) #file path main.py
    workPath = os.path.split(mainPath) #path working folder (whole file project)
    imagePath = "data_kain"

    data = Data(workPath, imagePath)
    trainSet, trainLabel, testSet, testLabel = data.load()

    """
    train image shape  (batch, channel, height, width)
    convert target class [0,1,2]
    one hot encoding
    [[1,0,0]
     [0,1,0]
     [0,0,1]]
    """

    Train_target = np.zeros((len(trainLabel),data.jum_kelas))
    Train_target[np.arange(len(trainLabel)), trainLabel[range(0, len(trainLabel))]] = 1
    
    Test_target = np.zeros((len(testLabel),data.jum_kelas))
    Train_target[np.arange(len(testLabel)), trainLabel[range(0, len(testLabel))]] = 1

    print("\nValidation set: ", testSet.shape, Test_target.shape)
    print("Training set: ", trainSet.shape, Train_target.shape)

    ### Create LeNet5 object ###
    mylenet = LENET5(trainSet, Train_target, testSet, Train_target)
    
    start = timeit.default_timer()
    mylenet.lenet_train(method="adam", epochs=4, batch=10, alpha=0.001, zeta=0)
    stop = timeit.default_timer()

    print("Training time:", stop - start)
    ### Save kernel and bias of conv and fc layers ###
    #mylenet.save_parameters()

    