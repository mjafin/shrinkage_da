# -*- coding: utf-8 -*-
"""
Shrinkage Estimation of Variance Vector, Correlation Matrix,
and Covariance Matrix 

@author Miika Ahdesmaki, Korbinian Strimmer
"""
from __future__ import print_function, division
import numpy as np
from sys import exit
from pvt_cppowscor import pvt_cppowscor
from pvt_svar import pvt_svar

def var_shrink(x, lambda_var = None, w = None, verbose = False):
    """Variance shrinkage

    Parameters
    ----------
    x : matrix array
        Data matrix with samples in rows, variables in columns.
    lambda_var : float
        Variance shrinkage parameter.
    verbose : bool
        Print out messages.
    
    Returns
    -------
    tuple
        tuple vs (array of variances), lambda_var (float), lambda_var_estimated (bool)
    
    """
    return pvt_svar(x, lambda_var, w, verbose)

def crossprod_powcor_shrink(x, y, alpha, lambda_cor = None, w = None, verbose=False):
    """computes R_shrink^alpha matrix-times y without expanding the correlation
    matrix (which can be huge)
    
    Parameters
    ----------
    x : matrix array
        Data matrix with samples in rows, variables in columns.
    y : vector array
        Vector(s), e.g. centroids, that are to be correlation adjusted (Mahalanobis).
    alpha : float
        Matrix power.
    lambda_cor : float
        Correlation shrinkage parameter.
    verbose : bool
        Print out messages.
    
    Returns
    -------
    vector array
        The result of the multiplication(s).
        
    """
    n, p = x.shape
    if y.shape[0] != p:
        exit("Input matrix/vector y must have p rows matching the number of columns in matrix x")
    return pvt_cppowscor(x, y, alpha, lambda_cor, w, verbose)