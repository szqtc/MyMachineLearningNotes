import numpy as np
from matplotlib import pyplot as plt
import networkx as nx

def gain(y_before, ys_after):
    """
    maximize it to find the best split
    
    Parameters
    ------------
    y_before : numpy array, (m, 1)
        the labels before splitted
    ys_after : list of numpy array, [(m1, 1), (m2, 1), ...], , m1+m2+...=m
        the list of labels splitted
    
    Returns
    ---------
    gain : scalar
        information gain
    """
    m = y_before.shape[0]
    #ys = set(y_before[:, 0].tolist())
    ys = np.unique(y_before)

    H0 = 0.
    for yval in ys:
        py = (y_before==yval).sum()/m
        H0 += -py*np.log2(py)

    H1 = 0.
    for y in ys_after:
        mx = y.shape[0]
        px = mx/m
        eps = 1./mx
        for yval in ys:
            pyx = (y==yval).sum()/mx
            if pyx < eps: continue
            H1 += px*(-pyx*np.log2(pyx))
            
    return H0-H1

def gain_ratio(y_before, ys_after):
    """
    maximize it to find the best split
    
    Parameters
    ------------
    y_before : numpy array, (m, 1)
        the labels before splitted
    ys_after : list of numpy array, [(m1, 1), (m2, 1), ...], , m1+m2+...=m
        the list of labels splitted
    
    Returns
    ---------
    gain_ratio : scalar
        information gain ratio
    """
    m = y_before.shape[0]
    #ys = set(y_before[:, 0].tolist())
    ys = np.unique(y_before)
    
    H0 = 0.
    for yval in ys:
        py = (y_before==yval).sum()/m
        H0 += -py*np.log2(py)

    H1, IV = 0., 0.
    for y in ys_after:
        mx = y.shape[0]
        px = mx/m
        if mx < 1: continue
        IV += -px*np.log2(px)
        eps = 1./mx
        for yval in ys:
            pyx = (y==yval).sum()/mx
            if pyx < eps: continue
            H1 += px*(-pyx*np.log2(pyx))

    if IV < 1e-7: IV = 1e-7
    return (H0-H1)/IV
    
def neg_gini_index(y_before, ys_after):
    """
    maximize it to find the best split
    
    Parameters
    ------------
    y_before : numpy array, (m, 1)
        the labels before splitted
    ys_after : list of numpy array, [(m1, 1), (m2, 1), ...], m1+m2+...=m
        the list of labels splitted
    
    Returns
    ---------
    neg_gini_index : scalar
        negative Gini index
    """
    m = y_before.shape[0]
    #ys = set(y_before[:, 0].tolist())
    ys = np.unique(y_before)
    
    G = 0.
    for y in ys_after:
        mx = y.shape[0]
        if mx < 1: continue
        px = mx/m
        for yval in ys:
            py = (y==yval).sum()/mx
            G += px*(1.-py**2)
    return -G

def neg_square_error(y_before, ys_after):
    """
    Parameters
    ------------
    y_before : numpy array, (m, 1)
        the labels before splitted
        *NOTE*: THIS PARAMETER IS REDUNDANT, WHICH
                      AIMS TO KEEP THE API CONSISTENT
    ys_after : list of numpy array, [(m1, 1), (m2, 1), ...], m1+m2+...=m
        the list of labels splitted
    
    Returns
    ---------
    neg_square_error : scalar
        negative of the sum of square error for each component in ys_after
    """
    sqerr = 0.
    for y in ys_after:
        ymean = y.mean()
        sqerr += ((y-ymean)**2).sum()
    return -sqerr


#### ID3, C4.5
class decision_tree:
    
    "Decision tree algorithm: ID3, C4.5"
    
    def __init__(self, scorefunc, maxdepth=None, seed=None):
        """
        Parameters
        -------------
        scorefunc : function, f(y: np.array, ys_new: list) -> float
            the function used to calculate the score of
            each feature
        maxdepth : positive integer, or None (default)
            the maximum depth of the tree
        seed : int or None
            random seed
        """
        self.scorefunc_ = scorefunc
        self.maxdepth_ = inf if maxdepth is None else maxdepth

        self.tree_ = None
        
        self.random_state = np.random.RandomState(seed)        

    @staticmethod
    def split_attribute(X, y, attr_axis):
        """
        Parameters
        ------------
        X : numpy array, (m, n)
            features of training examples
        y : numpy array, (m, 1)
            labels of training examples
        attr_axis : integer, [0, n)
            the index of attribute to split

        Returns
        ---------
        x_split : numpy array
            the split point of X in attr_axis
        Xs_new : list of numpy array, [(m1, n), (m2, n), ...]
            the freatures of splitted examples
        ys_new : list of numpy array, [(m1, 1), (m2, 1), ...]
            the label of splitted examples
        """
        xs = np.unique(X[:, attr_axis])
        Xs_new, ys_new = [], []
        for x in xs:
            filter_ = (X[:, attr_axis]==x)
            Xs_new.append(X[filter_, :])
            ys_new.append(y[filter_, :])
        return xs, Xs_new, ys_new
    
    def _feature_finder(self, X, y, filter_):
        """
        Parameters
        ------------
        X : numpy array, (m, n)
            features of training examples
        y : numpy array, (m, 1)
            labels of training examples

        Returns
        ---------
        best_index : integer, [0, n)
            the index of highest score feature
        best_score : double
            the score of that feature
        best_xsplit : numpy array
            the split point of X for the highest score feature
        best_Xs_new : list of numpy array, [(m1, n), (m2, n), ...]
            the features of splitted examples
        best_ys_new : list of numpy array, [(m1, 1), (m2, 1), ...]
            the label of splitted examples
        """
        # generate random indices
        n = X.shape[1]
        indices = np.arange(n)[filter_]
        self.random_state.shuffle(indices)

        #print('scores:')
        # find the best feature
        best_score = -inf
        best_index, best_xsplit = None, None
        best_Xs_new, best_ys_new = None, None
        for i in indices:
            xsplit, Xs_new, ys_new = self.split_attribute(X, y, i)
            score = self.scorefunc_(y, ys_new)
            #print(i, score)
            if score > best_score:
                best_score = score
                best_index = i
                best_xsplit = xsplit
                best_Xs_new = Xs_new
                best_ys_new = ys_new

        # the best feature will not be chosen again
        assert best_index is not None
        filter_[best_index] = False
        return best_index, best_score, best_xsplit, best_Xs_new, best_ys_new
    
    def _generate_tree(self, X, y, filter_, depth=0):
        """
        The main structure of generate a decision tree:
        1. if same type labels of features, set to be a leaf node
        2. find the best feature to separate, set to be a inner node
        3. separate the training data with the feature, deal
            with each of them recursively

        Parameters
        ------------
        X : numpy array, (m, n)
            features of training examples
        y : numpy array, (m, 1)
            labels of training examples
        filter_ : array of bools (n, )
            if the corresponding term of feature in filter_ is
            False, the feature will not be splitted
            *NOTE*: make sure not all False in the array
        depth : integer
            the depth of the subtree

        Returns
        ---------
        tree : dict
            {cond1 : subtree1, cond2 : subtree2, ...} for tree:
                           root
                cond1 /     \ cond2
                         /         \
               subtree1    subtree2
        """
        m, n = X.shape
        tree = {}

        # count the numbers of y
        ycounts = {}
        for yval in y.flat:
            if not yval in ycounts:
                ycounts[yval] = 1
            else:
                ycounts[yval] += 1

        # find the most frequent one
        maxcount = -1
        yfreq = None
        for k, v in ycounts.items():
            if v > maxcount:
                yfreq = k
                maxcount = v

        # default case: the most frequent label
        tree[None] = (None, yfreq) # axis, label

        # case 0: all the case with the same label
        if ycounts[yfreq] == m:
            #print('c0')
            return tree

        # case 1: reach the maxdepth
        if depth >= self.maxdepth_:
            #print('c1')
            return tree

        # case 2: no attribute remains
        if not filter_.any():
            #print('c2')
            return tree

        # case 3: all the features are the same
        i = 0
        while i<n and (not filter_[i] or (X[:, i]==X[0, i]).all()):
            i += 1
        if i == n:
            #print('c3')
            return tree

        # find the best feature
        best_index, _, best_xsplit, best_Xs_new, best_ys_new = self._feature_finder(X, y, filter_)

        # recursively generate the trees
        tree[None] = (best_index, yfreq) # axis, label
        for xsplit, X_new, y_new in zip(best_xsplit, best_Xs_new, best_ys_new):
            child = self._generate_tree(X_new, y_new, filter_=filter_.copy(), depth=depth+1)
            tree[xsplit] = child
        return tree
    
    def fit(self, X, y, filter_=None):
        """
        Fit a new tree from the given data
        
        Parameters
        ------------
        X : numpy array, (m, n)
            features of training examples
        y : numpy array, (m, 1)
            labels of training examples
        filter_ : array of bools (n, ), or None
            if the corresponding term of feature in filter_ is
            False, the feature will not be splitted
            *NOTE*: make sure not all False in the array
        """
        if filter_ is None:
            filter_ = np.array([True]*X.shape[1], dtype=bool)
        self.tree_ = self._generate_tree(X, y, filter_=filter_)

    def _predict_one(self, x):
        """
        Parameters
        -------------
        x : numpy array, (n, )
            a feature vector for a particular sample

        Returns
        ---------
        y : object
            the label of the input sample
        """
        assert self.tree_ is not None

        cur = self.tree_
        while cur[None][0] is not None:
            xval = x[cur[None][0]]
            if xval in cur:
                cur = cur[xval]
            else:
                break
        return cur[None][1]
    
    def predict(self, X):
        """
        Parameters
        -------------
        X : numpy array, (m, n)
            a feature vector for samples

        Returns
        ---------
        y : numpy array, (m, 1)
            the label of the input samples
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        m = X.shape[0]
        y = []
        for i in range(m):
            x = X[i, :]
            y.append(self._predict_one(x))
        return np.array(y).reshape(-1, 1)
    
    def visualize(self, colnames=None, figsize=(17, 8)):
        """
        Parameters
        -------------
        colnames : list, or None (default)
            the label of the column

        Returns
        --------
        fg : matplotlib.figure.Figure
            the figure of the decision tree
        """
        g = nx.Graph()

        node_labels, edge_labels = {}, {}
        next_index = 0
        def dfs(subtree, parent_index=None):
            nonlocal node_labels, edge_labels, next_index

            this_index = next_index
            g.add_node(this_index)
            if parent_index is not None:
                g.add_edge(parent_index, this_index)

            misc = subtree[None]
            node_labels[this_index] = misc[1]
            next_index += 1

            for key, dict_ in subtree.items():
                if key is None: continue
                if colnames is None:
                    edge_labels[(this_index, next_index)] = 'A[{}] is {}'.format(misc[0], key)
                else:
                    edge_labels[(this_index, next_index)] = '{} is {}'.format(colnames[misc[0]], key)
                dfs(dict_, this_index)

        dfs(self.tree_)

        fg, ax = plt.subplots(figsize=figsize)
        pos = nx.drawing.nx_agraph.graphviz_layout(g, prog='dot')
        nx.draw_networkx(g, pos, ax=ax, arrows=True, node_color='0.7', node_size=400,
            with_labels=True, labels=node_labels, font_size=14, font_color='k'
            )
        nx.draw_networkx_edge_labels(g, pos, ax=ax, edge_labels=edge_labels, font_size=11)

        ax.axis('off')
        return fg
    
    
# bottom up pruning
class pruning_mixin:

    def _prune(self, tree_dict, X_cross, y_cross):
        """
        bottom up pruning, only for classifier
        
        Parameters
        ------------
        tree_dict : dict
        X_cross : numpy array, (m', n)
            cross validation data
        y_cross : numpy array, (m', 1)
            cross validation labels

        Returns
        --------
        accuracy : double
            accuracy after pruning
        """
        misc = tree_dict[None]
        m = y_cross.shape[0]

        acc_postcut = (y_cross==misc[1]).sum()/m
        if misc[0] is None: #leaf
            return acc_postcut

        xsplits, Xs_new, ys_new = self.split_attribute(X_cross, y_cross, misc[0])
        n_correct = 0.
        for xsplit, X_new, y_new in zip(xsplits, Xs_new, ys_new):
            if xsplit in tree_dict:
                n_correct += self._prune(tree_dict[xsplit], X_new, y_new)*y_new.shape[0]
            else:
                n_correct += (y_new==misc[1]).sum() # outside the tree branch, use the majority label
        acc_precut = n_correct/m

        # start to prune
        if acc_precut < acc_postcut:
            for key in list(tree_dict.keys()):
                if key is not None:
                    del tree_dict[key]
            return acc_postcut
        return acc_precut
        
    def prune(self, X_cross, y_cross):
        """
        prune the tree in place

        Parameters
        ------------
        X_cross : numpy array, (m', n)
            cross validation data
        y_cross : numpy array, (m', 1)
            cross validation labels
        """
        self._prune(self.tree_, X_cross, y_cross)
        

class decision_tree2(decision_tree, pruning_mixin):
    """
    Decision Tree (ID3, C4.5) + pruning
    """
    pass


#### CART
def find_best_split_point(X, y, attr_axis, scorefunc):
    """
    Parameters
    ------------
    X : numpy array, (m, n)
        features of training examples
    y : numpy array, (m, 1)
        labels of training examples
    attr_axis : integer, [0, n)
        the index of attribute to split
    scorefunc : function, f(y: np.array, ys_new: list) -> float
        the function used to calculate the score of
        each feature

    Returns
    ---------
    x_split : double
        the best split value for the feature
    Xs_new : list of numpy array, [(m1, n), (m-m1, n)]
        the features of splitted examples
    ys_new : list of numpy array, [(m1, 1), (m-m1, 1)]
        the label of splitted examples
    best_score : double
    """
    x = np.unique(X[:, attr_axis]) # sorted
    if len(x) == 1:
        xmids = x
    else:
        xmids = (x[1:]+x[:-1])/2.
    
    best_score = -inf
    best_x_split, best_ys_new = None, None
    for xmid in xmids:
        filter_ = (X[:, attr_axis]<xmid)
        ys_new = [y[filter_, :], y[~filter_, :]]
        score = scorefunc(y, ys_new)
        #print(xmid, score)
        if score > best_score:
            best_score = score
            best_x_split = xmid
            best_ys_new = ys_new
    #print('>>', best_score, best_x_split)

    filter_ = (X[:, attr_axis]<best_x_split)
    best_Xs_new = [X[filter_, :], X[~filter_, :]]
    return best_x_split, best_Xs_new, best_ys_new, best_score


class cart_classifier(decision_tree):
    "CART classifier"

    split_attribute = None  # do not support pruning yet

    def __init__(self, scorefunc=neg_gini_index, maxdepth=None, seed=None):
        """
        Parameters
        -------------
        scorefunc : function, f(y: np.array, ys_new: list) -> float
            the function used to calculate the score of
            each feature
        maxdepth : positive integer, or None (default)
            the maximum depth of the tree
        seed : int or None
            random seed
        """
        super().__init__(scorefunc=scorefunc, maxdepth=maxdepth, seed=seed)

    def _feature_finder(self, X, y, filter_):
        """
        Parameters
        ------------
        X : numpy array, (m, n)
            features of training examples
        y : numpy array, (m, 1)
            labels of training examples
        filter_ : array of bools (n, )
            if the corresponding term of feature in filter_ is
            False, the feature will not be splitted
            *NOTE*: make sure not all False in the array

        Returns
        ---------
        best_index : integer, [0, n)
            the index of highest score feature
        best_score : double
            the score of that feature
        best_xsplit : numpy array
            the split point of X for the highest score feature
        best_Xs_new : list of numpy array, [(m1, n), (m2, n), ...]
            the freatures of splitted examples
        best_ys_new : list of numpy array, [(m1, 1), (m2, 1), ...]
            the label of splitted examples
        """
        # generate random indices
        n = X.shape[1]
        indices = np.arange(n)[filter_]
        self.random_state.shuffle(indices)

        # find the best feature
        best_score = -inf
        best_index, best_xsplit = None, None
        best_Xs_new, best_ys_new = None, None
        #print('score:')
        for i in indices:
            # do not split the feature with all the values the same
            if (X[:, i] == X[0, i]).all():
                #print('skip {}'.format(i), X[:, i], X[0, i])
                continue

            xsplit, Xs_new, ys_new, score = find_best_split_point(X, y, attr_axis=i, scorefunc=self.scorefunc_)
            #print('    ', i, score)
            if score > best_score:
                best_score = score
                best_index = i
                best_xsplit = xsplit
                best_Xs_new = Xs_new
                best_ys_new = ys_new

        #print(X, y)
        assert best_index is not None
        best_xsplit_out = ['<{}'.format(best_xsplit), '>={}'.format(best_xsplit)]
        #print(best_score, best_index, best_xsplit)
        # the best feature can be chosen again in CART
        return best_index, best_score, best_xsplit_out, best_Xs_new, best_ys_new

    def _predict_one(self, x):
        """
        Parameters
        -------------
        x : numpy array, (n, )
            a feature vector for a particular sample

        Returns
        ---------
        y : object
            the prediction of the input sample
        """
        cur = self.tree_
        while cur[None][0] is not None:
            if len(cur) == 1:
                break
            xval = x[cur[None][0]]
            for k, v in cur.items():
                if k is None:
                    continue
                elif k.startswith('<'):
                    thres = float(k[1:])
                    if xval < thres:
                        cur = cur[k]
                    else:
                        cur = cur['>='+k[1:]]
        return cur[None][1]

    def predict(self, X):
        """
        Parameters
        -------------
        X : numpy array, (m, n)
            a feature vector for samples

        Returns
        ---------
        y : numpy array, (m, 1)
            the prediction of the input samples
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        m = X.shape[0]
        y = []
        for i in range(m):
            y.append(self._predict_one(X[i, :]))
        return np.array(y).reshape(-1, 1)
    

class cart_regressor(cart_classifier):
    
    def __init__(self, scorefunc=neg_square_error, maxdepth=None, seed=None):
        """
        Parameters
        -------------
        scorefunc : function, f(y: np.array, ys_new: list) -> float
            the function used to calculate the score of
            each feature
        maxdepth : positive integer, or None (default)
            the maximum depth of the tree
        seed : int or None
            random seed
        """
        super().__init__(scorefunc=scorefunc, maxdepth=maxdepth, seed=seed)
        
    def _generate_tree(self, X, y, filter_, depth=0):
        """
        The main structure of generate a decision tree:
        1. if same type labels of features, set to be a leaf node
        2. find the best feature to separate, set to be a inner node
        3. separate the training data with the feature, deal
            with each of them recursively

        Parameters
        ------------
        X : numpy array, (m, n)
            features of training examples
        y : numpy array, (m, 1)
            labels of training examples
        scorefunc : function, f(y: np.array, ys_new: list) -> float
            the function used to calculate the score of
            each feature
        depth : positive integer
            the depth of the subtree
        filter_ : array of bools (n, )
            if the corresponding term of feature in filter_ is
            False, the feature will not be splitted

        Returns
        ---------
        tree : dict
            {cond1 : subtree1, cond2 : subtree2, ...} for tree:
                           root
                cond1 /     \ cond2
                         /         \
               subtree1    subtree2
        """
        m, n = X.shape
        tree = {}

        # default case: the average y value (different here)
        ydef = y.mean()
        tree[None] = (None, ydef) # axis, label

        # case 0: all the case with the same label
        if (y[:, 0] == y[0,0]).all():
            #print('c0')
            return tree

        # case 1: reach the maxdepth
        if depth >= self.maxdepth_:
            #print('c1')
            return tree

        # case 2: no attribute remains
        if not filter_.any():
            #print('c2')
            return tree

        # case 3: all the features are the same
        i = 0
        while i<n and (not filter_[i] or (X[:, i]==X[0, i]).all()):
            i += 1
        if i == n:
            #print('c3')
            return tree

        # find the best feature
        best_index, _, best_xsplit, best_Xs_new, best_ys_new = self._feature_finder(X, y, filter_)

        # recursively generate the trees
        tree[None] = (best_index, ydef) # axis, label
        for xsplit, X_new, y_new in zip(best_xsplit, best_Xs_new, best_ys_new):
            child = self._generate_tree(X_new, y_new, filter_=filter_.copy(), depth=depth+1)
            tree[xsplit] = child
        return tree
    

class random_cart_classifier(cart_classifier):
    """
    CART classifier. When do feature split, random feature will be chosen. Only used in random forest
    """
    
    def __init__(self, scorefunc=neg_gini_index, maxdepth=None, max_features=None, seed=None):
        """
        Parameters
        -------------
        scorefunc : function, f(y: np.array, ys_new: list) -> float
            the function used to calculate the score of
            each feature
        maxdepth : positive integer, or None (default)
            the maximum depth of the tree
        max_features : positive integer, or None
            for None, max_feature=sqrt(n_features)
        seed : int or None
            random seed
        """
        super().__init__(scorefunc=scorefunc, maxdepth=maxdepth, seed=seed)
        self.max_features = max_features
        
    def fit(self, X, y):
        """
        Parameters
        -------------
        X : numpy array, (m, n)
            input feature vectors
        y : numpy array, (m, 1)
            input labels
        """
        n = X.shape[1]
        if self.max_features is None:
            if n < 4:
                self.max_features = 1
            else:
                self.max_features = int(np.log2(n))
        filter_ = np.array([True]*self.max_features + [False]*(n-self.max_features))
        super().fit(X, y, filter_=filter_)
        
    def _generate_tree(self, X, y, filter_, depth=0):
        """
        Parameters
        ------------
        X : numpy array, (m, n)
            features of training examples
        y : numpy array, (m, 1)
            labels of training examples
        filter_ : array of bools (n, )
            if the corresponding term of feature in filter_ is
            False, the feature will not be splitted
            *NOTE*: make sure not all False in the array
        depth : integer
            the depth of the subtree
            
        Returns
        ---------
        tree : dict
        """
        self.random_state.shuffle(filter_) # choose random features
        return super()._generate_tree(X, y, filter_, depth=depth)