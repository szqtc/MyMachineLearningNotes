import numpy as np

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
    "Random forest"
    
    def __init__(self, estimator, n_estimators=10, max_features=None, oob_score=True, seed=None, **kwds):
        """
        Parameters
        -------------
        estimator : object
            
        n_estimator : integer, >0
            the number of decision tree
        max_features : positive integer, or None
            for None, max_feature=sqrt(n_features)
        seed : integer, or None
            the random seed
        oob_score : bool
            whether to calculate the out-of-bag
            score
        **kwds :
            the parameters passing to the estimator
        """
        self.max_features = max_features
        self.features_ = None
        
        super().__init__(estimator, n_estimators=n_estimators, oob_score=oob_score, seed=seed, **kwds)
        
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
        # select feature
        feat = list(range(X.shape[1]))
        self.random_state.shuffle(feat)
        feat = feat[:self.max_features]
        self.features_.append(feat)
        
        # shuffle the data and fit
        return super()._fit_one(X[:, feat], y, estimator)
        
    def fit(self, X, y):
        """
        Parameters
        -------------
        X : numpy array, (m, n)
            input feature vectors
        y : numpy array, (m, 1)
            input labels
        """
        if self.max_features is None:
            n = X.shape[0]
            if n < 4:
                self.max_features = 1
            else:
                self.max_features = int(np.log2(n))
        self.features_ = []
            
        super().fit(X, y)
        
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
        return self.estimators[i].predict(x[self.features_[i]]).ravel()[0]