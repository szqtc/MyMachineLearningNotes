import numpy as np
import tree

def bootstrap(X, y, size, random_state=None):
    """
    Bootrap sampling of the data
    
    Parameters
    -------------
    X : numpy array, (m, n)
        input feature vectors
    y : numpy array, (m, 1)
        input labels
    size : integer, >0
        the size of the resampled data
    random_state : np.random.RandomState object or None
        
    Returns
    ---------
    X_res : numpy array, (size, n)
        resampled feature vectors
    y_res : numpy array, (size, 1)
        resampled labels
    index : numpy array, (size, )
        the index of the resampled data
    """
    if random_state is None:
        new_idx = np.random.choice(X.shape[0], size=size)
    else:
        new_idx = random_state.choice(X.shape[0], size=size)
    return X[new_idx, :], y[new_idx], new_idx

class bagging:
    """
    Bagging
    """
    
    def __init__(self, estimator, n_estimators=10, oob_score=True, seed=None, **kwds):
        """
        Parameters
        -------------
        estimator : object
            
        n_estimator : integer, >0
            the number of decision tree
        seed : integer, or None
            the random seed
        oob_score : bool
            whether to calculate the out-of-bag
            score
        **kwds :
            the parameters passing to the estimator
        """
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.random_state = np.random.RandomState(seed)
        
        if 'seed' in kwds:
            del kwds[seed]
        
        self.estimators = []
        for i in range(self.n_estimators):
            varnames = estimator.__init__.__code__.co_varnames
            if varnames is not None and 'seed' in varnames:
                est = estimator(seed=self.random_state.randint(1000000000), **kwds)
            else:
                est = estimator(**kwds)
            self.estimators.append(est)
            
    def _fit_one(self, X, y, estimator):
        """
        Parameters
        -------------
        X : numpy array, (m, n)
            input feature vectors
        y : numpy array, (m, 1)
            input labels
        estimator : object
            estimate object
            
        Returns
        ---------
        oob_index : numpy array, or None
        oob_pred : numpy array, or None
        """
        m = X.shape[0]
        X_new, y_new, idx = bootstrap(X, y, m, random_state=self.random_state)
        estimator.fit(X_new, y_new)

        # collect the prediction on the oob data
        oob_index, oob_pred = None, None
        if self.oob_score:
            oob_index = list(set(range(m)) - set(idx.tolist()))
            oob_pred = estimator.predict(X[oob_index, :])
        return oob_index, oob_pred
    
    def fit(self, X, y):
        """
        Parameters
        -------------
        X : numpy array, (m, n)
            input feature vectors
        y : numpy array, (m, 1)
            input labels
        """
        m = X.shape[0]
        
        # the oob labels
        oob_data = []
        if self.oob_score:
            oob_data = [{} for _ in range(m)]
            
        for estimator in self.estimators:
            oob_index, oob_pred = self._fit_one(X, y, estimator)
            
            # collect the prediction on the oob data
            if self.oob_score:
                for i, v in zip(oob_index, oob_pred[:, 0]):
                    if v in oob_data[i]:
                        oob_data[i][v] += 1
                    else:
                        oob_data[i][v] = 1

        if self.oob_score:
            # evaluate the oob score
            n_sample, n_correct = 0., 0.
            for i, oob_count in enumerate(oob_data):
                y_true = y[i]
                maxcount = -1
                y_pred = None
                keys = list(oob_count.keys())
                self.random_state.shuffle(keys)
                for key in keys:
                    val = oob_count[key]
                    if val > maxcount:
                        maxcount = val
                        y_pred = key
                        
                if maxcount > 0:
                    n_sample += 1
                    if y_true == y_pred:
                        n_correct += 1
                    #print(y_true, y_pred, maxcount, oob_count)
            
            if n_sample > 0:
                self.oob_score_ = n_correct / n_sample
            else:
                self.oob_score_ = None
        else:
            self.oob_score_ = None
            
    def _predict_one(self, x, i):
        """
        Predict one input vector with i-th estimator
        
        Parameters
        -------------
        x : numpy array, (m, 1)
            input feature vector
        i : non-negative integer
            the index of the estimator
            
        Returns
        ---------
        y : object
            predicted value
        """
        return self.estimators[i].predict(x).ravel()[0]
    
    def predict(self, X):
        """
        Parameters
        -------------
        X : numpy array, (m', n)
            input feature vectors
            
        Returns
        ---------
        y_pred : numpy array, (m', 1)
            the predicted labels
        """
        labels = []
        for i in range(X.shape[0]):
            count = {}
            for j in range(self.n_estimators):
                l = self._predict_one(X[i, :], j)
                if l in count:
                    count[l] += 1
                else:
                    count[l] = 1
                    
            # find the most frequent one
            maxcount = -1
            yfreq = None
            keys = list(count.keys())
            self.random_state.shuffle(keys)
            for k in keys:
                v = count[k]
                if v > maxcount:
                    maxcount = v
                    yfreq = k
            labels.append(yfreq)
        return np.array(labels).reshape(-1, 1)


class random_forest(bagging):
    "random forest"
    
    def __init__(self, estimator=tree.random_cart_classifier, n_estimators=10, max_features=None, oob_score=True, seed=None, **kwds):
        """
        Parameters
        -------------
        estimator : object
            
        n_estimator : integer, >0
            the number of decision tree
        seed : integer, or None
            the random seed
        max_features : positive integer, or None
            for None, max_feature=sqrt(n_features)
        oob_score : bool
            whether to calculate the out-of-bag
            score
        **kwds :
            the parameters passing to the estimator
        """
        super().__init__(estimator, n_estimators=n_estimators, max_features=max_features, oob_score=oob_score, seed=seed, **kwds)
    

class adaboost_classifier:
    "AdaBoost based on the resampling method"
    
    def __init__(self, estimator, n_estimators=10, seed=None, **kwds):
        """
        Parameters
        -------------
        estimator : object
        n_estimator : integer, >0
            the number of decision tree
        seed : integer, or None
            the random seed, it will be used to initiate
            estimators of random algorithms (e.g. decision
            tree) and do resampling
        **kwds :
            the parameters passing to the estimator
        """
        self.n_estimators = n_estimators
        self.random_state = np.random.RandomState(seed)
        
        if 'seed' in kwds:
            del kwds[seed]
        
        self.estimators = []
        for i in range(self.n_estimators):
            varnames = estimator.__init__.__code__.co_varnames
            if varnames is not None and 'seed' in varnames:
                est = estimator(seed=self.random_state.randint(1000000000), **kwds)
            else:
                est = estimator(**kwds)
            self.estimators.append(est)
            
        self.alphas_, self.weights_ = None, None
        self.classes_mean_ = 0.

    def resample(self, weight, size):
        """
        Parameters
        ----------
        weight : numpy array, (m, 1)
            the distribution of each item in the
            training data
        size : positive integer
            the size of resampled data set
        """
        cum = np.cumsum(weight)
        #print(cum)
        rands = self.random_state.rand(size)
        indexes = []
        for r in rands:
            # bisect
            lo, hi = 0, size
            while lo < hi:
                mid = (lo+hi)//2
                midval = cum[mid]
                if r < midval:
                    hi = mid
                else:
                    lo = mid+1
            #print(r, lo, hi, cum[lo], cum[hi])
            indexes.append(lo)
        return np.array(indexes)
            
    def fit(self, X, y, maxiter=10):
        """
        Parameters
        -------------
        X : numpy array, (m, n)
            input feature vectors
        y : numpy array, (m, 1)
            input labels
        maxiter : positive integer, default: 10
            the maximum number of trials for one base
            estimator
        """
        self.classes_ = np.unique(y) # sorted unique values
        if len(self.classes_) > 2:
            raise NotImplementedError('The algorithm can only be used in binary classification')
        self.classes_mean_ = sum(self.classes_)/len(self.classes_)
        
        m = y.shape[0]
        self.alphas_ = []
        self.weights_ = np.full((m, 1), fill_value=1./m)
        for i, est in enumerate(self.estimators):
            #print(self.weights_.ravel()))
            for _ in range(maxiter):
                # fit the model with the resampled data
                rand_idx = self.resample(self.weights_, m)
                X_, y_ = X[rand_idx, :], y[rand_idx, :]
                est.fit(X_, y_)
                
                # evaluate the error of all training data
                y_pred = est.predict(X)
                good_predict = (y_pred == y)
                err = 1. - self.weights_[good_predict].sum()
                if err < 0.5:
                    break
            else:
                raise RuntimeError('number of fitting failures exceeds the limit!')

            if err < 1e-10: # all correct, only use the last one
                self.alphas_ = [0.]*(i-1) + [1.]
                break
            alpha = 0.5*np.log((1.-err)/err)
            self.alphas_.append(alpha)
            #print(err, alpha)
            
            # use predicted values as if y in {-1, 1}
            self.weights_ *= np.where(good_predict, np.exp(-alpha), np.exp(alpha))
            self.weights_ /= self.weights_.sum()
            
    def predict(self, X):
        """
        Parameters
        -------------
        X : numpy array, (m', n)
            input feature vectors
            
        Returns
        -------
        y : numpy array, (m', 1)
            input labels
        """
        H = 0.
        for alpha, est in zip(self.alphas_, self.estimators):
            H += alpha*np.where(est.predict(X)<=self.classes_mean_, -1., 1.)
        return np.where(H<=0., self.classes_[0], self.classes_[1]) # self.classes_ are sorted
