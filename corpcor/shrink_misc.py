# -*- coding: utf-8 -*-
"""
Non-public functions used in the covariance shrinkage estimator

@author Miika Ahdesmaki, Korbinian Strimmer
"""
from __future__ import print_function, division
from sys import exit
import numpy as np

def pvt_check_w(w, n):
    """Check that the weight vector w sums up to 1
    
    Parameters
    ----------
    w : numpy vector/array
        Vector of weights for samples.
    n : int
        Number of samples.

    
    Returns
    -------
    array
        Scaled weight vector.
        
    """
    if w is None: # return equal weights
        w = np.ones((n,1))/n
    else:
        if w.shape[0] != n:
            exit("Weight vector has incompatible length")
        w = w if np.sum(w) == 1 else w / np.sum(w)
    return w

def minmax(x, my_min=0, my_max=1):
    """Restrict float to between a minimum and maximum value
    Parameters
    ----------
    x : float
        Value to be restricted between my_min and my_max
    my_min : float
        Minimum value accepted (default 0)
    my_max : float
        Maximum value accepted (default 1)
    
    Returns
    -------
    float
        Value between my_min...my_max
        
    """
    return min(max(x, my_min), my_max)
