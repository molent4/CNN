from mnist import MNIST
import os, sys
import numpy as np
from lenet5 import *
import timeit
from Data import Data

class LoadMNISTdata:
    """docstring for LoadMNISTdata."""
    lim = 256.0

    def __init__(self, data_path):
        self.path = data_path

    def loadData(self):
        mndata = MNIST(self.path)
        train_img, train_label = mndata.load_training()
        test_img, test_label = mndata.load_testing()
        self.train_img = np.asarray(train_img, dtype='float64') / LoadMNISTdata.lim
        self.train_label = np.asarray(train_label)
        self.test_img = np.asarray(test_img, dtype='float64') / LoadMNISTdata.lim
        self.test_label = np.asarray(test_label)

        print("train_img:", self.train_img.shape)
        print("train_label:", self.train_label.shape)
        print("test_img:", self.test_img.shape)
        print("test_label:", self.test_label.shape)


if __name__ == '__main__':
    mainPath = os.path.dirname(os.path.abspath(__file__)) #file path main.py
    workPath = os.path.split(mainPath) #path working folder (whole file project)
    imagePath = "data_kain"

    data = Data(workPath, imagePath)

    cwd = os.getcwd()
    dataset = LoadMNISTdata(cwd)
    dataset.loadData()
    N = 50000
    #small_data = dataset.train_img[range(100)]
    #X_train = small_data.reshape(100, 1, 28,28)
    print(dataset.train_img.shape)
    X_train = dataset.train_img[range(0, N)].reshape(N, 1, 28, 28)
    Y_train = np.zeros((N, 10))
    Y_train[np.arange(N), dataset.train_label[range(0, N)]] = 1

    M = 10000
    X_valid = dataset.train_img[N:].reshape(M, 1, 28, 28)
    Y_valid = np.zeros((M, 10))
    Y_valid[np.arange(M), dataset.train_label[N:]] = 1
    print("Validation set: ", X_valid.shape, Y_valid.shape)
    print("Training set: ", X_train.shape, Y_train.shape)

    ### Create LeNet5 object ###
    mylenet = LENET5(X_train, Y_train, X_valid, Y_valid)

    