from .BaseNode import Node
import numpy as np
from .Init import kaiming_normal, kaiming_uniform, zeros


class relu(Node):
    def __init__(self):
        super().__init__("relu")

    def forward(self, x):
        self.cache.append(x)
        return np.clip(x, 0, None)

    def backward(self, grad):
        return np.multiply(grad, self.cache[-1] > 0)


class Linear(Node):
    def __init__(self, indim, outdim):
        """
        初始化
        @param indim: 输入维度
        @param outdim: 输出维度
        """
        weight = kaiming_uniform(indim, outdim)
        bias = zeros(outdim)
        super().__init__("linear", weight,
                         bias)  # Store the parameters in self.params, i.e., weight = self.params[0], bias =
        # self.params[1]

    def forward(self, X):
        """
        X: input data, dim: (Batch_size, indim), where "Batch_size" is the number of data samples in a batch.
        return: output data, dim: (Batch_size, outdim)
        """
        self.cache = [X]  # Store input for use in backward pass
        return X.dot(self.params[0]) + self.params[1]  # X * W + b

    def backward(self, grad):
        """
        grad: gradient of output data, dim: (Batch_size, outdim)
        return: gradient of input data, dim: (Batch_size, indim)
        """
        X = self.cache[0]
        self.grad = [X.T.dot(grad), np.sum(grad, axis=0)]  # Gradients for weights and biases
        return grad.dot(self.params[0].T)  # Gradient with respect to input


class sigmoid(Node):
    def __init__(self):
        super().__init__("sigmoid")

    def forward(self, X):
        """
        X: input data, dim: (*)
        return: output data, dim: (*)
        """
        output = 1 / (1 + np.exp(-X))
        self.cache = [output]
        return output

    def backward(self, grad):
        """
        grad: gradient of output data, dim: (*)
        return: gradient of input data, dim: (*)
        """
        output = self.cache[0]
        sigmoid_grad = output * (1 - output)
        return grad * sigmoid_grad


class MSE(Node):
    def __init__(self):
        super().__init__("MSE")

    def forward(self, X, Y):
        """
        X: output of model, dim: (Batch_size, *)
        Y: target data, dim: (Batch_size, *)
        return: MSE loss, dim: (1)
        """
        self.cache = [X, Y]
        loss = np.mean((X - Y) ** 2, axis=1)
        return np.mean(loss)

    def backward(self):
        """
        return: gradient of input data, dim: (Batch_size, *)
        """
        X, Y = self.cache
        m, n = X.shape
        return 2 * (X - Y) / (m * n)
