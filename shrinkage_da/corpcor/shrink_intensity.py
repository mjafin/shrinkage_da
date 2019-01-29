#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Functions for computing shrinkage intensity

This file implemets the same functionality as the R corpcor package file 
shrink.intensity.R

@author: Miika Ahdesmaki, Korbinian Strimmer
"""

from __future__ import print_function, division
from sys import exit
from shrink_misc import pvt_check_w, minmax
from wt_scale import wt_scale
from fast_svd import fast_svd
import numpy as np

def estimate_lambda_var(x, w = None, verbose = False):
    """Estimate variance shrinkage intensity
    
    Parameters
    ----------
    x : numpy array
        Samples by variables array of input data.
    w : numpy vector/array
        Vector of weights for samples.
    
    Returns
    -------
    float
        Shrinkage intensity
        
    """
    n, p = x.shape # how many samples and variables
    if n < 3:
        # Note that scikit-learn check_estimator expects to see a specific message
        # such as n_samples = 1 (verbatim)
        raise ValueError("Sample size too small. n_samples = 1")
    w = pvt_check_w(w, n)
    # bias correction factors
    w2 = np.sum(w*w)       # for w=1/n this equals 1/n   where n=dim(xs)[1]
    h1 = 1/(1-w2)       # for w=1/n this equals the usual h1=n/(n-1)
    h1w2 = w2/(1-w2)    # for w=1/n this equals 1/(n-1)
    xc, _ = wt_scale(x, w, center=True, scale=False) # standardise data matrix
    # compute empirical variances 
    v = h1*(np.sum(w*np.power(xc,2), axis=0, keepdims=True))
    # compute shrinkage target
    target = np.median(v)
    if verbose:
        print("Estimating optimal shrinkage intensity lambda.var (variance vector): ")

    zz = np.power(xc,2)
    q1 = np.sum( zz * w, axis=0, keepdims=True )
    q2 = np.sum( np.power(zz,2) * w, axis=0, keepdims=True ) - np.power(q1,2)   
    numerator = np.sum( q2 )
    denominator = np.sum( np.power(q1 - target/h1, 2) )
    
    if denominator == 0: 
        lambda_var = 1
    else:
        lambda_var = minmax(numerator/denominator * h1w2)
 
    if verbose: 
        print(lambda_var)   
  
    return float(lambda_var)
    
def estimate_lambda(x, w = None, verbose = False):
    """Estimate correlation shrinkage intensity
    
    Parameters
    ----------
    x : numpy array
        Samples by variables array of input data.
    w : numpy vector/array
        Vector of weights for samples.
    
    Returns
    -------
    float
        Shrinkage intensity
        
    """
    # test data x = np.array([range(2,10),np.power(range(2,10),2),np.power(range(2,10),3)])
    # test data w = np.array([[0.5],[0.6],[0.7]])
    n, p = x.shape # how many samples and variables
    if p==1:
        return float(1)
    if n < 3:
        exit("Sample size too small!")
    w = pvt_check_w(w, n)
    xs, _ = wt_scale(x, w, center=True, scale=True) # standardise data matrix
    if verbose:
        print("Estimating optimal shrinkage intensity lambda (correlation matrix): ")
    # bias correction factors
    w2 = np.sum(w*w)           # for w=1/n this equals 1/n   where n=dim(xs)[1]
    h1w2 = w2/(1-w2)        # for w=1/n this equals 1/(n-1)

    sw = np.sqrt(w)
   
    # direct slow algorithm illustrated in R code
    #  E2R = (crossprod(sweep(xs, MARGIN=1, STATS=sw, FUN="*")))^2
    #  ER2 = crossprod(sweep(xs^2, MARGIN=1, STATS=sw, FUN="*"))
    #  ## offdiagonal sums
    #  sE2R = sum(E2R)-sum(diag(E2R))
    #  sER2 = sum(ER2)-sum(diag(ER2))
 
    # Here's how to compute off-diagonal sums much more efficiently for n << p
    # this algorithm is due to Miika Ahdesm\"aki
    xsw = xs * sw # numpy broadcast
    
    svd_d, svd_u, svd_v = fast_svd(xsw)
    sE2R = np.sum(xsw * np.matmul((svd_u*np.power(svd_d,3)), svd_v.T)) - np.sum(np.power(np.sum(np.power(xsw,2),axis=0, keepdims=True),2)) 
    xs2w = np.power(xs,2) * sw
    sER2 = 2*np.sum(xs2w[:,range(p-2,-1,-1)] * np.cumsum(xs2w[:,range(p-1,0,-1)], axis=1))
    #######
    denominator = sE2R
    numerator = sER2 - sE2R
    if denominator == 0:
        my_lambda = 1
    else:
        my_lambda = minmax(numerator/denominator * h1w2)

    if verbose:
        print(my_lambda)

    return float(my_lambda)