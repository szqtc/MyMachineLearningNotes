import numpy as np
from . import kernels

class linear_regressor:
    
    def __init__(self):
        self.w_, self.b_ = None, 0.
        
    def fit(self, X, y):
        """
        Parameters
        ------------
        X : numpy array, (m, n)
            features of training examples
        y : numpy array, (m, 1)
            labels of training examples
        """
        m, n = X.shape
        X_ = np.c_[np.ones((m, 1)), X]
        w = np.linalg.inv(X_.T@X_) @ X_.T @ y # @ for dot product
        self.w_ = w[1:]
        self.b_ = w[0].squeeze()
        
    def predict(self, X):
        """
        Parameters
        -------------
        X : numpy array, (m', n)
            feature vectors

        Returns
        ---------
        y : numpy array, (m', 1)
            predicted outputs
        """
        return X@self.w_+self.b_

class krr:
    "Kernel Ridge Regression"
    
    support_kernels = {'rbf': kernels.rbf, 'linear': kernels.linear, 'poly' : kernels.poly}
    
    def __init__(self, kernel='rbf', C=1., **kwds):
        """
        Parameters
        -------------
        kernel : string, ='rbf', 'linear', or others
            the kernel adopted
        C : scalar, >0.
            constant to control the regularization
        """
        self.C_ = C

        if kernel in self.support_kernels:
            self.kernel_ = kernel
            kernel_func = self.support_kernels[kernel]
        elif callable(kernel):
            self.kernel_ = kernel.__name__
            kernel_func = kernel
        else:
            raise ValueError('The kernel {} is not support now'.format(kernel))

        self.kernel_parameters_ = {}
        nargs = kernel_func.__code__.co_argcount
        nargs_default = 0 if kernel_func.__defaults__ is None else len(kernel_func.__defaults__)
        for i in range(2, nargs):
            var = kernel_func.__code__.co_varnames[i]
            if var in kwds:
                val = kwds[var]
            elif i >= nargs - nargs_default:
                val = kernel_func.__defaults__[i-(nargs - nargs_default)]
            else:
                raise TypeError('parameter {} is required'.format(var))
            self.kernel_parameters_[var] = val
        self.kernel_function_ = lambda x1, x2: kernel_func(x1, x2, **self.kernel_parameters_)
        
    def fit(self, X_data, y_data):
        """
        Parameters
        -------------
        X_data : (m, n) array
            training data
        y_data : (m, 1) array, =+/-1
            the label of training data
        """
        m, n = X_data.shape
        kXX = self.kernel_function_(X_data.reshape(m, 1, n), X_data.reshape(1, m, n)) # m x m
        mat = kXX + np.eye(m)/self.C_
        self.mat_ = np.linalg.inv(mat)@y_data # m, 1
        self.X_fit_ = X_data
        
    def predict(self, X):
        """
        Parameters
        -------------
        X : (m', n) array
            input features

        Returns
        ---------
        y_pred : (m', 1)
            predicted labels of input features
        """
        m1, n = X.shape
        kXx = self.kernel_function_(X.reshape(m1, 1, n), self.X_fit_.reshape(1, -1, n)) # m1, m
        return kXx@self.mat_ # m1, 1
    
