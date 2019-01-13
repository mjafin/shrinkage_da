# -*- coding: utf-8 -*-
"""
Efficient computation of crossprod(R^alpha, y)

@author Miika Ahdesmaki, Korbinian Strimmer
"""
from __future__ import print_function, division
import numpy as np
from scipy.linalg import fractional_matrix_power
from fast_svd import fast_svd
from wt_scale import wt_scale
from shrink_intensity import estimate_lambda
from shrink_misc import pvt_check_w, minmax
from sys import exit


def pvt_cppowscor(x, y, alpha, lambda_cor = None, w = None, verbose=False):
    """Private function estimating a correlation matrix product without explicitly
    evaluating the correlation matrix
    
    This procedure exploits a special identity to efficiently
    compute the crossprod of matrix power of the correlation shrinkage estimator with y
    
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
    -------Class
    dict
        The result of the multiplication(s) and correlation shrinkage parameter
    """
    if lambda_cor is None:
        lambda_cor = estimate_lambda(x = x, w = w, verbose = verbose)
    lambda_cor = minmax(lambda_cor) # make sure lambda isn't improper
    n, p = x.shape
    try:
        yn, yp = y.shape
    except ValueError:
        exit("Input y to pvt_cppowscor() must be a matrix of column vectors. Dimensionality may have dropped along the way when slicing.")
    if yn != p:
        exit("There is something wrong with the dimensionalities of y and x")
    w = pvt_check_w(w, n)
    xs, sc = wt_scale(x, w, center=True, scale=True) # standardise data matrix
    zeros = sc == 0
    w2 = np.sum(w*w)       # for w=1/n this equals 1/n   where n=dim(xs)[1]
    h1 = 1/(1-w2)       # for w=1/n this equals the usual h1=n/(n-1)
    if lambda_cor == 1 or alpha == 0: # in both cases R is the identity matrix
        return y
    (d, u, v) = fast_svd(xs)
    m = d.shape[0] # rank of xs
    d = np.column_stack(d).T # make d into a column vector
    UTWU = np.matmul(u.T, u * w) # U' matmul diag(w) matmul U
    C = UTWU * d * d.T # D matmul UTWU matmul D
    C = (1-lambda_cor) * h1 * C
    C = (C + C.T)/2  # symmetrise for numerical reasons
    # note: C is of size m x m, and diagonal if w=1/n
    if lambda_cor == 0: # use eigenvalue decomposition computing the matrix power
        cp_powr = np.matmul(v, np.matmul(fractional_matrix_power(C, alpha), np.matmul(v.T, y) ) )
    else:
        F = np.eye(m) - fractional_matrix_power(C/lambda_cor + np.eye(m), alpha)
        cp_powr = (y - np.matmul(v, np.matmul(F, np.matmul(v.T, y) ) )) * np.power(lambda_cor,alpha)
    # set all diagonal entries in R_shrink corresponding to zero-variance variables to 1
    cp_powr[zeros,:] = y[zeros,:]
    return dict(cp_powr = cp_powr, lambda_cor = lambda_cor)