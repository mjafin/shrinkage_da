# -*- coding: utf-8 -*-
"""
Weighted Expectations and Variances

@author Miika Ahdesmaki, Korbinian Strimmer
"""
from __future__ import print_function, division
from sys import exit
from shrink_misc import pvt_check_w
import numpy as np

def wt_var(x, w):
    """Estimate weighted variance for a vector
    
    Parameters
    ----------
    x : numpy array vector
        Sample vector.
    w : numpy vector/array
        Vector of weights for samples.
    
    Returns
    -------
    array
        Vector array of (weighted) variances
    """
    w = pvt_check_w(w, x.shape[0]) # x.shape[0] is number of samples
    h1 = 1/(1-sum(w*w))  # for w=1/n this equals the usual h1=n/(n-1)
    xc = x - np.average(x, weights = w)
    s2 = h1 * np.average(xc*xc, weights = w)
    return s2
    
def wt_moments(x, w):
    """Estimate weighted moments
    
    Parameters
    ----------
    x : numpy array
        Samples by variables array of input data.
    w : numpy vector/array
        Vector of weights for samples.
    center : bool
        Centre data
    scale : bool
        Scale data
    
    Returns
    -------
    array
        Centred and/or scaled matrix/array
        
    """
    w = pvt_check_w(w, x.shape[0]) # x.shape[0] is number of samples
    if not isinstance(x, np.ndarray):
        exit("Input x to wt_scale() must be numpy array")
    # bias correction factor
    h1 = 1/(1-sum(w*w))   # for w=1/n this equals the usual h1=n/(n-1)
 
     
    m = np.sum(w*x, axis = 0) # column sums
  
    v = h1*(np.sum(w*np.power(x, 2), axis = 0) - np.power(m,2))
 
  
    # set small values of variance exactly to zero
    v[v < np.finfo(float).eps] = 0
  
    return dict(mean=m, var=v)

def wt_scale(x, w, center=True, scale=True):
    """scale using weights    x : array
        The first parameter.
    w : column vector/array
        The second parameter.
apply(a, 2, wt.var, w=w
    
    Parameters
    ----------
    x : numpy array
        Samples by variables array of input data.
    w : numpy vector/array
        Vector of weights for samples.
    center : bool
        Centre data
    scale : bool
        Scale data
    
    Returns
    -------
    array
        Centred and/or scaled matrix/array
        
    """
    if not isinstance(x, np.ndarray):
        exit("Input x to wt_scale() must be numpy array")
    w = pvt_check_w(w, x.shape[0]) # x.shape[0] is number of samples
    # compute column means and variances
    wm = wt_moments(x, w)
    sc = None
    if center:
        # use numpy broadcasting to subtract means in vector from matrix 
        x = x - wm["mean"]
    if scale:
        sc = np.sqrt(wm["var"])
        sc[sc == 0] = np.inf # this help with division by zero
        # use numpy broadcasting to scale data
        x = x / sc # if mean is not subtracted, note that the mean will shift due to scaling

    return x, sc
    