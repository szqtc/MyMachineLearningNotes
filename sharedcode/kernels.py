import numpy as np

def rbf(x1, x2, gamma=1.):
    """
    rbf kernel function
    
    Parameters
    ------------
    x1 : numpy array, (..., n)
        the first input feature vector
    x2 : numpy array, (..., n)
        the second input feature vector
    gamma : positive double, default: 1
        gamma=0.5/sigma**2
    
    Returns
    ---------
    kernel : numpy array
        output kernel
    """
    return np.exp(-gamma*((x1-x2)**2).sum(axis=-1))

def linear(x1, x2):
    """
    linear kernel function
    
    Parameters
    ------------
    x1 : numpy array, (..., n)
        the first input feature vector
    x2 : numpy array, (..., n)
        the second input feature vector
    
    Returns
    ---------
    kernel : numpy array
        output kernel
    """
    return (x1*x2).sum(axis=-1)

def poly(x1, x2, degree=3, gamma=1., r=0.):
    """
    polynomial kernel function
    
    Parameters
    ------------
    x1 : numpy array, (..., n)
        the first input feature vector
    x2 : numpy array, (..., n)
        the second input feature vector
    degree : positive double, default: 3
        degree of the polynomial kernel function
    gamma : positive double, default: 1
        kernel coefficient
     r : positive double, default: 0
         independent term
    
    Returns
    ---------
    kernel : numpy array
        output kernel
    """
    return (gamma*(x1*x2).sum(axis=-1) + r)**degree