# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 12:47:32 2021

@author: ASUS
"""
import os
from conv import *
from relu import *
from sigmoid import *
from fc import *
from maxpool import *
from softmax import *

from loss import *
import timeit
from sklearn.metrics import accuracy_score

from tqdm import tqdm
from Data import Data




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
class LENET5:
    """docstring forLENET5."""
    def __init__(self, t_input, t_output, v_input, v_output):
        """
        Creates Lenet-5 architecture
        Input:
            t_input: True Training input of shape (N, Depth, Height, Width)
            t_output: True Training output of shape (N, Class_Number)
        """
        b_dir = "16/"
        """
        Convo Input:
            (layer_size, kernel_size, fan, **params)
            layer_size: tuple consisting (depth, height, width)
            kernel_size: tuple consisting (number_of_kernels, inp_depth, inp_height, inp_width)
            fan: tuple of number of nodes in previous layer and this layer
            params: directory consists of pad_len and stride,
                    filename (to load weights from file)
        """
        """
        kernel_size: tuple consisting (number_of_kernels, inp_depth, inp_height, inp_width)
        number of kernel = number of filter
        input dept is channel depth for first convo layer and number of prev kernel for convo 2
        fan (area input resolution, depth multiply by area of input)
        """
        # Conv Layer-1
        conv1 = CONV_LAYER((6, 48, 48), (6, 3, 5, 5), (2304, 13824), pad=2, stride=1,) #filename=b_dir+"conv0.npz")
        relu1 = RELU_LAYER()
        # Sub-sampling-1
        pool2 = MAX_POOL_LAYER(stride=2)
        # Conv Layer-2
        conv3 = CONV_LAYER((6, 24, 24), (6, 6, 3, 3), (576, 3456), pad=0, stride=1, )#filename=b_dir+"conv3.npz")
        relu3 = RELU_LAYER()
        # Sub-sampling-2
        pool4 = MAX_POOL_LAYER(stride=2)
        # Fully Connected-1
        fc5 = FC_LAYER(120, (726, 120), )#filename=b_dir+"fc6.npz")
        relu5 = SIGMOID_LAYER()
        # Fully Connected-2
        fc6 = FC_LAYER(84, (120, 84), )#filename=b_dir+"fc8.npz")
        relu6 = SIGMOID_LAYER()
        # Fully Connected-3
        output = FC_LAYER(4, (84, 4), )#filename=b_dir+"fc10.npz")
        softmax = SOFTMAX_LAYER()
        self.layers = [conv1, relu1, pool2, conv3, relu3, pool4, fc5, relu5, fc6, relu6, output, softmax]

        self.X = t_input
        self.Y = t_output
        self.Xv = v_input
        self.Yv = v_output

    @staticmethod
    def one_image_time(X, layers):
        """
        Computes time of conv and fc layers
        Input:
            X: Input
            layers: List of layers.
        Output:
            inp: Final output
        """
        inp = X
        conv_time = 0.0
        fc_time = 0.0
        layer_time = []

        for layer in layers:
            start = timeit.default_timer()
            if isinstance(layer, FC_LAYER) and len(inp.shape) == 4:
                inp, ws = layer.forward(inp.reshape(inp.shape[0], inp.shape[1]*inp.shape[2]*inp.shape[3]))
            else:
                inp, ws = layer.forward(inp)
            stop = timeit.default_timer()
            layer_time += [stop-start]
            if isinstance(layer, (FC_LAYER, SIGMOID_LAYER, SOFTMAX_LAYER)):
                fc_time += stop - start
            if isinstance(layer, (CONV_LAYER, RELU_LAYER)):
                conv_time += stop - start
        return conv_time, fc_time, layer_time


    @staticmethod
    def feedForward(X, layers):
        """
        Computes final output of neural network by passing
        output of one layer to another.
        Input:
            X: Input
            layers: List of layers.
        Output:
            inp: Final output
        """
        inp = X
        wsum = 0
        for layer in layers:
            if isinstance(layer, FC_LAYER) and len(inp.shape) == 4:
                inp, ws = layer.forward(inp.reshape(inp.shape[0], inp.shape[1]*inp.shape[2]*inp.shape[3]))
            else:
                inp, ws = layer.forward(inp)
            wsum += ws
        return inp, wsum

    @staticmethod
    def backpropagation(Y, layers):
        """
        Computes final output of neural network by passing
        output of one layer to another.
        Input:
            Y: True output
            layers: List of layers.
        Output:
            grad: gradient
        """
        delta = Y
        for layer in layers[::-1]:
            delta = layer.backward(delta)

    @staticmethod
    def update_parameters(layers, batch_size, a, z, m):
        """
        Update weight parameters of each layer
        """
        for layer in layers:
            if isinstance(layer, (CONV_LAYER, FC_LAYER)):
                layer.update_kernel(batch=batch_size, alpha=a, zeta=z, method=m)

    @staticmethod
    def loss_function(pred, t, **params):
        """
        Computes loss using cross-entropy method.
        Input:
            pred: Predicted output of network of shape (N, C)
            t: true output of shape (N, C)
            w_sum: sum of squares of all weight parameters for L2 regularization
        where,
            N: batch size
            C: Number of classes in the final layer
        Output:
            Loss or cost
        """
        w_sum = params.get("wsum", 0)
        #print("w_sum: ", w_sum)
        z = params.get("zeta", 0)
        #assert t.shape == pred.shape
        #print("Shape: ", t.shape, pred.shape,z)
        epsilon = 1e-10
        return ((-t * np.log(pred + epsilon)).sum() + (z/2)*w_sum) / pred.shape[0]


    def lenet_train(self, **params):
        """
        Train the Lenet-5.
        Input:
            params: parameters including "batch", "alpha"(learning rate),
                    "zeta"(regularization parameter), "method" (gradient method),
                    "epochs", ...
        """
        batch  = params.get("batch", 50)             # Default 50
        alpha  = params.get("alpha", 0.01)            # Default 0.1
        zeta   = params.get("zeta", 0)               # Default 0 (No regularization)
        method = params.get("method", "adam")            # Default
        epochs = params.get("epochs", 10)             # Default 4
        print("Training on params: batch=", batch, " learning rate=", alpha, " L2 regularization=", zeta, " method=", method, " epochs=", epochs)
        self.loss_history = []
        self.gradient_history = []
        self.valid_loss_history = []
        self.valid_acc_history = []
        self.step_loss = []
        print(method)
        X_train = self.X
        Y_train = self.Y
        assert X_train.shape[0] == Y_train.shape[0]
        num_batches = int(np.ceil(X_train.shape[0] / batch))
        step = 0;
        steps = []
        X_batches = zip(np.array_split(X_train, num_batches, axis=0), np.array_split(Y_train, num_batches, axis=0))

        for ep in range(epochs):
            print("Epoch: ", ep, "===============================================")
            for x, y in X_batches:
                predictions, weight_sum = LENET5.feedForward(x, self.layers)
                if len(y.shape) == 2 and len(predictions.shape) == 2 :
                    temp_y = np.argmax(y, axis=1)
                    temp_predictions = np.argmax(predictions, axis=1)
                    accuracy = np.sum(np.equal(temp_y,temp_predictions))/len(temp_y)
                    """
                    # np.argmax return indexs of max value each row (axis 1)
                    # np.argmax return array of index refering to position of maximun value along axis 1
                    """
                else :
                    accuracy = np.mean(predictions==y)
             
                loss = LENET5.loss_function(predictions, y, wsum=weight_sum, zeta=zeta)
                self.loss_history += [loss]
                LENET5.backpropagation(y, self.layers)          #check this gradient
                LENET5.update_parameters(self.layers, x.shape[0], alpha, zeta, method)
                print("Step: ", step, ":: Loss: ", loss, "weight_sum: ", weight_sum, "acc", accuracy)
                if step % 10 == 0:
                    pred, w = LENET5.feedForward(self.Xv, self.layers)
                    v_loss = LENET5.loss_function(pred, self.Yv, wsum=w, zeta=zeta)
                    print("Validation error: ", v_loss)
                    if len(pred.shape) == 2 and len(self.Yv.shape) == 2 :
                        temp_y = np.argmax(self.Yv, axis=1)
                        temp_predictions = np.argmax(pred, axis=1)
                        v_acc = np.sum(np.equal(temp_y,temp_predictions))/len(temp_y)
                        print("Validation acc: ", v_acc)
                    else :
                        v_acc = np.mean(pred==yV)
                    steps += [step]
                    self.valid_loss_history += [v_loss]
                    self.valid_acc_history += [v_acc]
                    self.step_loss += [loss]
                step += 1

            XY = list(zip(X_train, Y_train))
            np.random.shuffle(XY)
            new_X, new_Y = zip(*XY)
            assert len(new_X) == X_train.shape[0] and len(new_Y) == len(new_X)
            X_batches = zip(np.array_split(new_X, num_batches, axis=0), np.array_split(new_Y, num_batches, axis=0))
        np.savez("step_loss_history", self.step_loss, self.valid_loss_history)
        np.savez("loss_history", self.loss_history)
        LENET5.plots(self.loss_history, self.step_loss, self.valid_loss_history, steps)
        pass

    def lenet_predictions(self, X, Y):
        """
        Predicts the ouput and computes the accuracy on the dataset provided.
        Input:
            X: Input of shape (Num, depth, height, width)
            Y: True output of shape (Num, Classes)
        """
        start = timeit.default_timer()
        predictions, weight_sum = LENET5.feedForward(X, self.layers)
        stop = timeit.default_timer()

        loss = LENET5.loss_function(predictions, Y, wsum=weight_sum, zeta=0.99)
        y_true = np.argmax(Y, axis=1)
        y_pred = np.argmax(predictions, axis=1)
        print("Dataset accuracy: ", accuracy_score(y_true, y_pred)*100)
        print("FeedForward time:", stop - start)
        pass

    @staticmethod
    def plots(x, y, z, steps):
        try:
            plt.figure(1)
            plt.plot(x, '-bo', label="Loss")
            plt.xlabel('Number of iterations', fontsize=18)
            plt.ylabel('Loss', fontsize=18)
            plt.title('Training Error rate vs Number of iterations')
            plt.savefig("Loss_function_vs_iter.jpeg")
        except:
            pass

        try:
            plt.figure(2)
            plt.plot(steps, y, '-bo', label="Training Loss")
            plt.plot(steps, z, '-ro', label="Validation Loss")
            plt.xlabel('Number of iterations', fontsize=18)
            plt.ylabel('Loss Value', fontsize=18)
            plt.title('Training and Validation error rates vs number of iterations')
            plt.legend(loc='upper right')
            plt.savefig("error_rates.jpeg")
        except:
            pass
        pass
    
    def save_parameters(self):
        """
        Saves the weights and biases of Conv and Fc layers in a file.
        """
        for layer in self.layers:
            if isinstance(layer, CONV_LAYER):
                np.savez("conv" + str(self.layers.index(layer)), layer.kernel, layer.bias)
            elif isinstance(layer, FC_LAYER):
                np.savez("fc" + str(self.layers.index(layer)), layer.kernel, layer.bias)
        pass

    def check_gradient(self):
        """
        Computes the numerical gradient and compares with Analytical gradient
        """
        sample = 10
        epsilon = 1e-4
        X_sample = self.X[range(sample)]
        Y_sample = self.Y[range(sample)]
        predictions, weight_sum = LENET5.feedForward(X_sample, self.layers)
        LENET5.backpropagation(Y_sample, self.layers)

        abs_diff = 0
        abs_sum = 0

        for layer in self.layers:
            if not isinstance(layer, (CONV_LAYER, FC_LAYER)):
                continue
            i = 0
            print("\n\n\n\n\n")
            print(type(layer))
            del_k = layer.delta_K + (0.99*layer.kernel/sample)
            kb = chain(np.nditer(layer.kernel, op_flags=['readwrite']), np.nditer(layer.bias, op_flags=['readwrite']))
            del_kb = chain(np.nditer(del_k, op_flags=['readonly']), np.nditer(layer.delta_b, op_flags=['readonly']))

            for w, dw in zip(kb, del_kb):
                w += epsilon
                pred, w_sum = LENET5.feedForward(X_sample, self.layers)
                loss_plus = LENET5.loss_function(pred, Y_sample, wsum=w_sum, zeta=0.99)

                w -= 2*epsilon
                pred, w_sum = LENET5.feedForward(X_sample, self.layers)
                loss_minus = LENET5.loss_function(pred, Y_sample, wsum=w_sum, zeta=0.99)

                w += epsilon
                numerical_gradient = (loss_plus - loss_minus)/(2*epsilon)

                abs_diff += np.square(numerical_gradient - dw)
                abs_sum  += np.square(numerical_gradient + dw)
                print(i, "Numerical Gradient: ", numerical_gradient, "Analytical Gradient: ", dw)
                if not np.isclose(numerical_gradient, dw, atol=1e-4):
                    print("Not so close")
                if i >= 10:
                    break
                i += 1

        print("Relative difference: ", np.sqrt(abs_diff)/np.sqrt(abs_sum))
        pass
    
def main():
    mainPath = os.path.dirname(os.path.abspath(__file__)) #file path main.py
    workPath = os.path.split(mainPath) #path working folder (whole file project)
    imagePath = "data_kain"
    
    layer_time = []
    data = Data(workPath, imagePath)
    X_train, trainLabel, X_test, testLabel = data.load()
    kelas = data.jum_kelas
    len_label = trainLabel.shape[0]
    
    Y_train = np.zeros((len_label,kelas))
    Y_train[np.arange(len_label), trainLabel[range(0, len_label)]] = 1
    
    kelas = data.jum_kelas
    len_label = testLabel.shape[0]
    Y_test = np.zeros((len_label, kelas))
    Y_test[np.arange(len_label), testLabel[range(0, len_label)]] = 1
    
    mylenet = LENET5(X_train, Y_train, X_test, Y_test)
    
    start = timeit.default_timer()
    mylenet.lenet_train(method="adam", epochs=500, batch=64, alpha=0.001, zeta=0)
    stop = timeit.default_timer()
    
    print("Training time:", stop - start)
        ### Save kernel and bias of conv and fc layers ###
    mylenet.save_parameters()
    
        ### Training Set accuracy ###
    print("Training ", end="")
    mylenet.lenet_predictions(X_train, Y_train)

if __name__=='__main__':
    main()

    

