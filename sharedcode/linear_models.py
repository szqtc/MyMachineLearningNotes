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


class ridge_regressor(linear_regressor):
    "Linear model with regularization"
    
    def __init__(self, C=1.):
        """
        Parameters
        -------------
        C : non-negative double
            regularization factor
        """
        self.C_ = C
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
        I = np.diag([0.]+[1.]*n) # do not regularize the interception
        w_ = np.linalg.inv(X_.T @ X_ + I/self.C_) @ X_.T @ y
        self.w_ = w_[1:]
        self.b_ = w_[0].squeeze()


class kernel_ridge_regressor:
    "Kernel ridge regression"
    
    support_kernels = {'rbf': rbf, 'linear': linear, 'poly' : poly}
    
    def __init__(self, kernel='rbf', C=1., **kwds):
        """
        Parameters
        -------------
        kernel : string, ='rbf', 'linear', or others
            the kernel adopted
        C : scalar, >0.
            constant to control the regularization
        **kwds :
            parameters passing to the kernel function
        """
        self.C_ = C

        # set up kernels
        if kernel in self.support_kernels:
            self.kernel_ = kernel
            kernel_func = self.support_kernels[kernel]
        elif callable(kernel):
            try:
                self.kernel_ = kernel.__name__
            except AttributeError:
                self.kernel_ = str(kernel)
            kernel_func = kernel
        else:
            raise ValueError('The kernel {} is not support now'.format(kernel))

        self.kernel_parameters_ = kwds.copy()
        self.kernel_function_ = lambda x1, x2: kernel_func(x1, x2, **self.kernel_parameters_)
        
    def fit(self, X_data, y_data):
        """
        Parameters
        -------------
        X_data : (m, n) array
            training data
        y_data : (m, 1) array, =+/-1
            the label of training data
            
        *NOTE*: we assume interception=0.
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
    

class locally_weighted_linear_regressor:
    "Locally weighted linear regression"
    
    def __init__(self, tau=1.):
        """
        Parameters
        -------------
        tau : float, >0., default: 1.
            the sigma in the Gaussian distribution
        """
        self.tau_ = tau
        self.X_, self.y_ = None, None
        
    def fit(self, X, y):
        """
        Parameters
        -------------
        X_train : numpy array, (m, n)
            features of the training sample
        y_train : numpy array, (m, 1)
            output of the training sample
        """
        m = X.shape[0]
        self.X_ = np.c_[np.ones((m, 1)), X]
        self.y_ = y.copy()
        
    def predict(self, X):
        """
        Parameters
        -------------
        X : numpy array, (m', n)
            the input feature to infer

        Returns
        ---------
        y_pred : numpy array, (m', 1)
            the predicted output of the given feature
        """
        mm = X.shape[0]
        X_ = np.c_[np.ones((mm, 1)), X]
        
        y_pred = []
        for i in range(mm):
            sqdist = ((self.X_ - X_[i:i+1, :])**2).sum(axis=1)
            M = np.diag(np.exp(-sqdist/2./self.tau_**2))
            w = np.linalg.inv(self.X_.T@M@self.X_) @ self.X_.T @ M @ self.y_
            y_pred.append(X_[i, :]@w)
        return np.array(y_pred).reshape(-1, 1)


class logistic_regressor:
    "Logistic regression with gradient descent"
    
    def __init__(self, saveloss=True):
        """
        Parameters
        -------------
        saveloss : bool, default: True
            whether to save the loss
        """
        self.saveloss = saveloss
        self.losses = []
        
        self.w_, self.b_ = None, 0.
        self.learning_rate_, self.nepoch_ = None, None
    
    @staticmethod
    def loss(y, pprob):
        """
        The loss function of logistic regression
        
        Parameters
        -------------
        y : numpy array, (m, 1)
            training target
        pprob : numpy array, (m, 1)
            the probabilty of positive type
            
        Returns
        ---------
        loss : double
        """
        return -(y*np.log(pprob) + (1-y)*np.log(1.-pprob)).sum()
    
    def fit(self, X, y, learning_rate=0.1, nepoch=1000):
        """
        Parameters
        -------------
        X_train : numpy array, (m, n)
            features of the training sample
        y_train : numpy array, (m, 1)
            output of the training sample
        learning_rate : positive double, default: 0.1
        nepoch : positive integer
        """
        self.learning_rate_ = learning_rate
        self.nepoch_ = nepoch

        # gradient descent
        self.w_ = np.zeros((X.shape[1], 1))
        self.b_ = 0.
        self.losses = []
        for _ in range(nepoch):
            p = self.predict_proba(X)
            dy = (y-p)*self.learning_rate_
            self.b_ += dy.sum()
            self.w_ += X.T@dy

            if self.saveloss:
                self.losses.append(self.loss(y, p))
    
    def predict_proba(self, X):
        """
        Parameters
        ------------
        X : (m, n) array
            m training samples with n features

        Returns
        ---------
        probability : (m, 1) array
            predicted probability for the positive case
        """
        return 1./(1.+np.exp(-X@self.w_ - self.b_))
    
    def predict(self, X):
        """
        Parameters
        ------------
        X : (m, n) array
            m training samples with n features

        Returns
        ---------
        y_pred : (m, 1) array
            predicted labels
        """
        return self.predict_proba(X)>0.5
    

class softmax_regressor:
    "Softmax regression"
    
    def __init__(self, C=1., saveloss=True):
        """
        Parameters
        -------------
        C : non-negative double, default: 1
            regularization factor
        saveloss : bool, default: True
            whether to save the loss
        """
        self.C_ = C
        
        self.W_ = None
        self.learning_rate_ = None
        self.nepoch = None

        self.saveloss = saveloss
        self.losses =[]
        
    def predict_proba(self, X):
        """
        Parameters
        -------------
        X : numpy array, (m, n)
            input features

        Returns
        ---------
        y_proba : numpy array, (m, t)
            predicted probablity in each class
        """
        Z = X@self.W_[1:, :] + self.W_[:1, :]
        expZ = np.exp(Z)
        denorm = 1.+expZ.sum(axis=1, keepdims=True)
        return np.c_[expZ/denorm, 1./denorm]
        
    def predict(self, X):
        """
        Parameters
        ------------
        X : numpy array, (m, n)
            input features

        Returns
        ---------
        y : numpy array, (m, 1)
            the best matched labels
        """
        return self.predict_proba(X).argmax(axis=1).reshape(-1, 1)
        
    @staticmethod
    def loss(y, yprob):
        """
        Parameters
        -------------
        y : integer numpy array, (m, 1)
            input labels
        yprob : numpy array, (m, t)
            the probality of each types

        Returns
        ---------
        loss : double
        """
        t = yprob.shape[1]
        loss = 0.
        for i in range(t):
            probs = yprob[y[:, 0]==i, i]
            loss -= np.log(np.maximum(probs, 1e-10)).sum() # to avoid prob=0.
        return loss
        
    def fit(self, X, y, learning_rate=0.1, nepoch=1000):
        """
        Parameters
        -------------
        X_train : numpy array, (m, n)
            features of the training sample
        y_train : numpy array, (m, 1)
            output of the training sample
        learning_rate : positive double, default: 0.1
        nepoch : positive integer
        """
        self.learning_rate_ = learning_rate
        self.nepoch = nepoch

        m, n = X.shape
        X_ = np.c_[np.ones(m), X]
        
        t = np.unique(y).shape[0] # num of output types
        self.W_ = np.zeros((n+1, t-1)) # +1 for b

        # gradient descent
        self.losses = []
        for i in range(nepoch):
            p = self.predict_proba(X)
            for j in range(t-1):
                self.W_[:, j] += (X_.T@((y[:, 0]==j) - p[:, j])*self.C_ - self.W_[:, j])*self.learning_rate_
                
            if self.saveloss:
                self.losses.append(self.loss(y, p))