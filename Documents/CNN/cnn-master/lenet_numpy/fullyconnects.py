import numpy as np
import sys
class FC_LAYER:
    """docstring forRELU_LAYER."""
    def __init__(self, layer_size, kernel_size, **params):
        """
        Input:
            layer_size: number of neurons/nodes in fc layer
            kernel: kernel of shape (nodes_l1 , nodes_l2)
            fan: tuple of number of nodes in previous layer and this layer
        """
        self.nodes = layer_size

        fname = params.get("filename", None)
        if fname:
            try:
                arr_files = np.load(fname)
                self.kernel = arr_files['arr_0']
                self.bias = arr_files['arr_1']
                assert np.all(self.kernel.shape == kernel_size) and np.all(self.bias.shape[0] == kernel_size[1])
            except:
                 raise
        else:
            f = np.sqrt(6)/np.sqrt(kernel_size[0] + kernel_size[1])
            epsilon = 1e-6
            self.kernel = np.random.uniform(-f, f + epsilon, kernel_size)
            self.bias = np.random.uniform(-f, f + epsilon, kernel_size[1])
        self.gradient_history = np.zeros(kernel_size)
        self.bias_history = np.zeros(kernel_size[1])
        self.m_kernel = np.zeros(kernel_size)
        self.m_bias = np.zeros(kernel_size[1])
        self.v_kernel = np.zeros(kernel_size)
        self.v_bias = np.zeros(kernel_size[1])
        self.timestamp = 0
        pass

    def forward(self, X):
        """
        Computes the forward pass of Sigmoid Layer.
        Input:
            X: Input data of shape (N, nodes_l1)
        Variables:
            kernel: Weight array of shape (nodes_l1, nodes_l2)
            bias: Biases of shape (nodes_l2)
        where,
            nodes_l1: number of nodes in previous layer
            nodes_l2: number of nodes in this fc layer
        """
        kernel, bias = self.kernel, self.bias
        self.cache = (X, kernel, bias)
        self.output = np.dot(X, kernel) + bias
        #assert self.activations.shape == (X.shape[0], bias.shape[0])
        return self.output, np.sum(np.square(self.kernel))

    def backward(self, delta):
        """
        Computes the backward pass of Sigmoid Layer.
        Input:
            delta: Shape of delta values (N, nodes_l2)
        """
        X, kernel, bias = self.cache
        self.delta_X = np.dot(delta, kernel.T)
        self.delta_K = np.dot(X.T, delta)
        #print(self.delta_K[0][range(10)], self.delta_K.shape)
        #print(X.T[0][range(10)], X.T.shape)
        #print(delta[0][range(10)], delta.shape)
        self.delta_b = np.sum(delta, axis=0)
        return self.delta_X

