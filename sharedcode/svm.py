import numpy as np
from . import kernels

inf = float('inf')

class linearSVC:
    """
    Linear SVC
    """
    
    def __init__(self, C=1., tol=1e-3, **kwds):
        """
        Parameters
        -------------
        C : scalar, >0.
            constant to control the regularization
        tol : scalar, >0, default: 1e-3
            tolerence of the fitting
        """
        assert C > 0. and tol > 0.
        self.C_, self.tol_ = C, tol
        self.b_ = 0.
        
        self.support_vectors_ = None
        self.dual_coef_ = None

        self._init_model(**kwds)
        
    def _init_model(self):
        """
        Initialized Parameters
        ------------------------
        w_ : (n, 1) array
            weight vector
        b_ : scalar
            bias
        """
        self.w_, self.b_ = None, 0.
    
    def __call__(self, X):
        """
        evaluate SVC model
        
        Parameters
        -------------
        X : (m', n) array
            input features

        Returns
        ---------
        y_pred : (m', 1)
            predicted labels of input features
        """
        return self.predict(X)

    def predict(self, X):
        """
        Linear SVC model
        
        Parameters
        -------------
        X : (m', n) array
            input features

        Returns
        ---------
        y_pred : (m', 1)
            predicted labels of input features
        """
        assert self.w_ is not None
        return X@self.w_ + self.b_ # np.dot(X, w) + b

    @staticmethod
    def support_vector_indices(alpha):
        """
        Find the indices of support vectors, which satisfies
        \alpha_i != 0. due to the KKT condition.

        Parameters
        -------------
        alpha : (m, 1) array, 0<=alpha<=C
            dual parameters

        Returns
        ---------
        indices : array of non-negative integers
            the index of support vectors in the data
        """
        return np.nonzero(alpha>1.e-7)[0]

    def update_model(self, alpha, X_data, y_data):
        """
        Update the bias and weight terms of the model.
        
        Parameters
        -------------
        alpha : (m, 1) array, 0<=alpha<=C
            dual parameters
        X_data : (m, n) array
            training data
        y_data : (m, 1) array, =+/-1
            the label of training data
        """
        idxs = self.support_vector_indices(alpha)
        X_sv, y_sv = X_data[idxs, :], y_data[idxs, :]
        self.support_vectors_ = X_sv
        self.dual_coef_ = y_sv*alpha[idxs, :]

        self.w_ = (self.dual_coef_*self.support_vectors_).sum(axis=0, keepdims=True).T
        self.b_ = sum(1./y_sv - self.support_vectors_@self.w_) /y_sv.shape[0]

    def eta(self, xi, xj):
        """
        Calculate eta=|| \Phi(xi) - \Phi(xj) ||^2
        """
        return ((xi-xj)**2).sum()
    
    def update_alpha(self, alpha, i, j, X_data, y_data):
        """
        Update the dual parameters based on SMO algorithm

        Parameters
        -------------
        alpha : (m, 1) array, >=0.
            dual parameters
        i, j : integer, >=0, i != j
            the term of alpha needed to be update
        X_data : (m, n) array
            training data
        y_data : (m, 1) array, =+/-1
            the label of training data

        Returns
        ---------
        is_updated : bool
            whether the update is performed
        """
        if i<0 or j<0 or i==j:
            return False

        # calculate errors for i, j
        xi, yi = X_data[i:i+1, :], y_data[i, 0] # xi in (1, n), yi is scalar
        yi_pred = self.predict(xi) # b is canceled out, so here we just use b=0
        Ei = yi_pred - yi

        xj, yj = X_data[j:j+1, :], y_data[j, 0]
        yj_pred = self.predict(xj)
        Ej = yj_pred - yj

        # Compute L, H
        alpha_i, alpha_j = alpha[i, 0], alpha[j, 0]
        if yi*yj < 0:
            L = max(0., alpha_i-alpha_j)
            H = min(self.C_, self.C_+alpha_i-alpha_j)
        else:
            L = max(0., alpha_i+alpha_j-self.C_)
            H = min(self.C_, alpha_i+alpha_j)
        if L >= H:
            #print("[skipped] L >= H")
            return False

        # compute eta
        eta = self.eta(xi, xj)
        if eta <= 0.:
            #print('[skipped] eta <= 0')
            return False # ignored since it rarely happens
        
        # new alphas
        dalpha_i = yi/eta * (Ej - Ei)
        alpha_i_new = np.clip(alpha_i+dalpha_i, a_min=L, a_max=H) # L <= alpha_new <= H
        if abs(alpha_i_new-alpha_i) <= 1e-6*(alpha_i_new + alpha_i + 1e-6):
            #print('[skipped] not enough update for alpha')
            return False
        alpha_j_new = (alpha_i-alpha_i_new)*yi*yj + alpha_j # a1y1 + a2y2 = a'1y1 + a'2y2, |yi|=1

        # update parameters
        alpha[i] = alpha_i_new
        alpha[j] = alpha_j_new
        self.update_model(alpha, X_data, y_data)
        return True
    
    def examine(self, i, alpha, X_data, y_data):
        """
        Do one-step fitting given the first index
        
        Parameters
        -------------
        i : integer, >=0
            the first index
        alpha : (m, 1) array, >=0.
            dual parameters
        X_data : (m, n) array
            training data
        y_data : (m, 1) array, =+/-1
            the label of training data
            
        Returns
        ---------
        is_fitted : bool
        """
        m = X_data.shape[0]
        y_pred = self.predict(X_data)
        
        ai= alpha[i, 0]
        ri = y_data[i, 0]*y_pred[i, 0]-1.
        
        # when KKT condition does not satisfy
        # KKT: 0<a<C, y*y_pred=1; a=0, y*y_pred>=1; a=C, y*y_pred<=1
        if (ai<self.C_ and ri<-self.tol_) or (ai>0 and ri>self.tol_):
            # Heuristic:
            # 1. nonbound points are more likely to be updated, while those
            #    at boundary are kept at the same position
            # 2. dalpha is larger when |E_i - E_j| is larger
            nonbds = set(np.nonzero((alpha>0) & (alpha<self.C_))[0])
            if nonbds:
                errs = y_pred-y_data
                j = np.abs(errs-errs[i]).argmax()
                if self.update_alpha(alpha, i, j, X_data, y_data):
                    return True

            for j in nonbds:
                if self.update_alpha(alpha, i, j, X_data, y_data):
                    return True
                
            for j in range(m):
                if not j in nonbds and self.update_alpha(alpha, i, j, X_data, y_data):
                    return True
        return False

    def _init_fit(self, n):
        """
        Initialize the parameter

        Parameters
        -------------
        n : integer
            number of featurs
        """
        self.w_ = np.zeros((n, 1))

    def fit(self, X_data, y_data):
        """
        Fit the SVC model with SMO algorithm
        
        Parameters
        -------------
        X_data : (m, n) array
            training data
        y_data : (m, 1) array, =+/-1
            the label of training data
            
        Returns
        ---------
        alpha : (m, 1) array, 0<=alpha<=C
            dual parameters
        """
        m, n = X_data.shape
        if y_data.ndim == 1:
            y_data = y_data.reshape((-1, 1))
        alpha = np.zeros((m, 1))
        self._init_fit(n)
        
        it = 0
        num_change = 0
        examine_all = True
        while (num_change>0 or examine_all):
            if examine_all:
                arridx = range(m) # find the points break KKT conditions
            else:
                arridx = np.nonzero((alpha>0)&(alpha<self.C_))[0]

            num_change = 0 # the number of pairs of changed alphas in this pass    
            for i in range(m):
                num_change += self.examine(i, alpha, X_data, y_data)

            if examine_all:
                examine_all = False
            elif num_change == 0:
                examine_all = True

            #print('iter={}, dual={}, loss={}'.format(it, self.dual_gain(), self.loss(X_data, y_data)))
            print('iter={}, dual={}'.format(it, self.dual_gain()))
            it += 1
        return alpha

    def dual_gain(self):
        """
        Calculate only with support vectors
        
        Returns
        --------
        gain : scalar
        """
        # more general case
        #ws = alpha*y_data*X_data
        #return alpha.sum() - 0.5*(ws@ws.T).sum()
        
        # only use support vectors
        ws = self.support_vectors_*self.dual_coef_
        return np.abs(self.dual_coef_).sum() - 0.5*(ws@ws.T).sum()

    def loss(self, X_data, y_data):
        """
        Parameters
        -------------
        X_data : (m, n) array
            training data
        y_data : (m, 1) array, =+/-1
            the label of training data

        Returns
        ---------
        loss : scalar
        """
        y_pred = self.predict(X_data)
        return 0.5*(self.w_**2).sum() + self.C_*np.maximum(0., 1.-y_data*y_pred).sum()
    

class kernelSVC(linearSVC):
    """
    Kernelized SVC
    """
    
    support_kernels = {'rbf': kernels.rbf, 'linear': kernels.linear, 'poly' : kernels.poly}

    def __init__(self, kernel='rbf', C=1., tol=1e-3, **kwds):
        """
        Parameters
        -------------
        kernel : string, ='rbf', 'linear', or others
            the kernel adopted
        C : scalar, >0.
            constant to control the regularization
        tol : scalar, >0, default: 1e-3
            tolerence of the fitting
        """
        super().__init__(C=C, tol=tol, kernel=kernel, **kwds)
    
    def _init_model(self, kernel='rbf', **kwds):
        """
        Parameters
        -------------
        kernel : string, ='rbf', 'linear', or function
            the kernel adopted

        Optional Parameters
        -----------------------
        gamma : positive double
            for rbf kernel, gamma=0.5/sigma**2
        """
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
        for i in range(2, kernel_func.__code__.co_argcount):
            var = kernel_func.__code__.co_varnames[i]
            if var in kwds:
                val = kwds[var]
            elif i >= nargs - nargs_default:
                val = kernel_func.__defaults__[i-(nargs - nargs_default)]
            else:
                raise TypeError('parameter {} is required'.format(var))
            self.kernel_parameters_[var] = val
        self.kernel_function_ = lambda x1, x2: kernel_func(x1, x2, **self.kernel_parameters_)

    def update_model(self, alpha, X_data, y_data):
        """
        Update the bias and weight terms of the model.
        
        Parameters
        -------------
        alpha : (m, 1) array, 0<=alpha<=C
            dual parameters
        X_data : (m, n) array
            training data
        y_data : (m, 1) array, =+/-1
            the label of training data
        """
        idxs = self.support_vector_indices(alpha)
        X_sv, y_sv = X_data[idxs, :], y_data[idxs, :]
        self.support_vectors_ = X_sv
        self.dual_coef_ = y_sv*alpha[idxs, :]
        
        n_support = len(idxs)
        K = self.kernel_function_(X_sv.reshape(1, n_support, -1), X_sv.reshape(n_support, 1, -1))
        self.b_ = (sum(1./y_sv) - (self.dual_coef_.reshape(1, n_support) *K).sum()) /y_sv.shape[0]

    def predict(self, X):
        """
        kernelized SVC model
        
        Parameters
        -------------
        X : (m', n) array
            input features

        Returns
        ---------
        y_pred : (m', 1)
            predicted labels of input features
        """
        if X.ndim == 1:
            n_sample = 1
        else:
            n_sample = X.shape[0]
        n_support = self.dual_coef_.shape[0]
        #print(n_sample, n_support)
        K = self.kernel_function_(self.support_vectors_.reshape(1, n_support, -1), X.reshape(n_sample, 1, -1))
        y_pred = (self.dual_coef_.reshape(1, n_support) *K).sum(axis=-1, keepdims=True) + self.b_
        #print(y_pred.shape, K.shape)
        return y_pred
    
    def _init_fit(self, n):
        self.dual_coef_ = np.array([[0.]])
        self.support_vectors_ = np.array([[0.]*n])

    def eta(self, xi, xj):
        """
        Calculate eta=|| \Phi(xi) - \Phi(xj) ||^2
        """
        kii = self.kernel_function_(xi, xi)
        kjj = self.kernel_function_(xj, xj)
        kij = self.kernel_function_(xi, xj)
        return kii+kjj-2.*kij

    def dual_gain(self):
        """
        Calculate only with support vectors
        
        Returns
        --------
        gain : scalar
        """
        #print(self.support_vectors_)
        nfeature = self.support_vectors_.shape[-1]
        K = self.kernel_function_(self.support_vectors_.reshape(-1, 1, nfeature), self.support_vectors_.reshape(1, -1, nfeature))
        return np.abs(self.dual_coef_).sum() - 0.5*(self.dual_coef_.reshape(-1, 1)*K*self.dual_coef_.reshape(1, -1)).sum()
    
    def loss(self, X_data, y_data):
        """
        Parameters
        -------------
        X_data : (m, n) array
            training data
        y_data : (m, 1) array, =+/-1
            the label of training data

        Returns
        ---------
        loss : scalar
        """
        nfeature = self.support_vectors_.shape[-1]
        y_pred = self.predict(X_data)
        K = self.kernel_function_(self.support_vectors_.reshape(-1, 1, nfeature), self.support_vectors_.reshape(1, -1, nfeature))
        return 0.5*(self.dual_coef_.reshape(-1, 1)*K*self.dual_coef_.reshape(1, -1)).sum() + \
                    self.C_*np.maximum(0., 1.-y_data*y_pred).sum()