import numpy as np


class Base_kernel:
    
    def __init__(self):
        pass
    
    def __call__(self, x1, x2):
        """
        Linear kernel function.
        
        Arguments:
            x1: shape (n1, d)
            x2: shape (n2, d)
            
        Returns:
            y : shape (n1, n2), where y[i, j] = kernel(x1[i], x2[j])
        """
        pass


class Linear_kernel(Base_kernel):
    
    def __init__(self):
        super().__init__()
    
    def __call__(self, x1, x2):
        # TODO: Implement the linear kernel function
        y = np.dot(x1, x2.T)
        return y
    
    
class Polynomial_kernel(Base_kernel):
        
    def __init__(self, degree, c):
        super().__init__()
        self.degree = degree
        self.c = c
        
    def __call__(self, x1, x2):
        # TODO: Implement the polynomial kernel function
        y = (np.dot(x1, x2.T) + self.c) ** self.degree
        return y


class RBF_kernel(Base_kernel):
    
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma 

    def __call__(self, x1, x2):
        # TODO: Implement the RBF kernel function
        if len(x1.shape) == 1:
            x1 = x1[np.newaxis, :]
        if len(x2.shape) == 1:
            x2 = x2[np.newaxis, :]
        x1_square = np.sum(np.square(x1), axis=1)
        x2_square = np.sum(np.square(x2), axis=1)
        square_norm = -2 * np.dot(x1, x2.T) + x1_square[:, None] + x2_square[None, :]
        y = np.exp(-square_norm / (2 * self.sigma ** 2))
        return y
