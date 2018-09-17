import numpy as np

def rss(y_pred, y_truth):
    """
    Residual sum of squares (RSS)
    
    Parameters
    -------------
    y_pred : numpy array, (m, 1)
        the predicted labels
    y_truth : numpy array, (m, 1)
        the true labels
        
    Returns
    --------
    rss : double
    """
    return ((y_pred-y_truth)**2).sum()

def mse(y_pred, y_truth):
    """
    mean squared error (MSE)

    Parameters
    -------------
    y_pred : numpy array, (m, 1)
        the predicted labels
    y_truth : numpy array, (m, 1)
        the true labels
        
    Returns
    --------
    mse : double
        smaller is better
    """
    return rss(y_pred, y_truth)/y_pred.shape[0]

def rse(y_pred, y_truth):
    """
    residual standard error (RSE):
        estimation of the standard deviation between the
        predicted values and true values

    REF: An Introduction to Statistical Learning, pp. 82-83

    Parameters
    -------------
    y_pred : numpy array, (m, 1), m>30
        the predicted labels
    y_truth : numpy array, (m, 1), m>30
        the true labels
        
    Returns
    --------
    rse : double
        smaller is better
    """
    return np.sqrt(rss(y_pred, y_truth)/(y_pred.shape[0]-2))

def r2(y_pred, y_truth):
    """
    R^2 statistic:
        the proportion of variance explained by the model
        given X
    
    Parameters
    -------------
    y_pred : numpy array, (m, 1)
        the predicted labels
    y_truth : numpy array, (m, 1)
        the true labels
        
    Returns
    --------
    r2 : double, [0, 1]
        larger is better
    """
    tss_ = ((y_truth-y_truth.mean(axis=0))**2).sum()
    rss_ = rss(y_pred, y_truth)
    return 1.-rss_/tss_

def accuracy(y_pred, y_truth):
    """
    Accuracy: #right / #all
    
    Parameters
    -------------
    y_pred : numpy array, (m, 1)
        the predicted labels
    y_truth : numpy array, (m, 1)
        the true labels
        
    Returns
    --------
    accuracy : double
        larger is better
    """
    return (y_pred == y_truth).sum()/y_truth.shape[0] # right/all

def precision(y_pred, y_truth):
    """
    Precision: #true_positive / #pred_positive
    
    Parameters
    -------------
    y_pred : numpy array, (m, 1)
        the predicted labels
    y_truth : numpy array, (m, 1)
        the true labels
        
    Returns
    --------
    precision : double
        larger is better
    """
    return ((y_pred==1) & (y_truth == 1)).sum()/(y_pred == 1).sum() # tp / pred pos

def recall(y_pred, y_truth):
    """
    Recall: #true_positive / #positive
    
    Parameters
    -------------
    y_pred : numpy array, (m, 1)
        the predicted labels
    y_truth : numpy array, (m, 1)
        the true labels
        
    Returns
    --------
    recall : double
        larger is better
    """
    return ((y_pred==1) & (y_truth == 1)).sum()/(y_truth == 1).sum() # tp / pos

def fbeta_score(y_pred, y_truth, beta=1.):
    """
    F_beta score

    Parameters
    -------------
    y_pred : numpy array, (m, 1)
        the predicted labels
    y_truth : numpy array, (m, 1)
        the true labels
    beta : double, >0. default: 1.
        
    Returns
    --------
    fbeta_score : double
        larger is better
    """
    prec = precision(y_pred, y_truth)
    reca = recall(y_pred, y_truth)
    return (1+beta**2)/(1/prec + beta**2/reca)