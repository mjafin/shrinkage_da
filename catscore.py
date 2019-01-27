# -*- coding: utf-8 -*-
"""
Estimate CAT scores and t-scores

@author Miika Ahdesmaki, Korbinian Strimmer
"""
from __future__ import print_function, division
import numpy as np
from sys import exit
from centroids import centroids
from corpcor.shrink_estimates import crossprod_powcor_shrink


def catscore(Xtrain, L, lambda_cor = None, lambda_var = None, lambda_freqs = None, diagonal=False, verbose=False):
    """Estimate CAT scores and t-scores
    
    Parameters
    ----------
    Xtrain : numpy array
        Samples-in-rows matrix.
    L : list
        Class labels in a list. Must match number of rows in Xtrain.
    lambda_cor : float
        Correlation shrinkage parameter.
    lambda_var : float or list
        Shrinkage parameter for variances or list specified_lambda_var = lambda_varif separate ones used per class
    lambda_freqs : float
        Shrinkage parameter for class prevalences.
    diagonal : bool
        If True, skip correlation adjustment and assume diagonal model (False)
    verbose : bool
        Verbose mode (False).
    
    Returns
    -------
    dictionary
        Dictionary containing (ca)t-scores, shrinkage parameters
    """
    nX, pX = Xtrain.shape
    if len(L) != nX:
        exit("Number of rows in input matrix Xtrain must match the number of class labels")
    regularisation = dict(lambda_cor = 1, lambda_var = np.nan, lambda_freqs = np.nan) # regularisation parameters for correlation, variance and priors
    my_cent = centroids(Xtrain, L, lambda_var, lambda_freqs, var_groups = False, centered_data = True, verbose = verbose)
    cl_count = len(my_cent["groups"]) - 1 # number of classes 
    n = np.sum(my_cent["samples"]) # number of samples
    p = my_cent["means"].shape[0] # number of features
    
    mu = my_cent["means"][:,range(0,cl_count)] # centroids
    mup = my_cent["means"][:,cl_count] # pooled centroid
    sc = np.sqrt(my_cent["variances"][:,0])
    regularisation["lambda_var"] = my_cent["var_lambdas"][0]
    
    #class frequencies
    freqs = my_cent["freqs"]
    regularisation["lambda_freqs"] = my_cent["freqs_lambda"]
    xc = my_cent["centered_data"]
    
    ############################################################# 
    # compute (correlation-adjusted) t-scores for each variable
    #############################################################

    cat = np.zeros((p, cl_count))
    # first compute t-scores (centroid vs. pooled mean)
  
    # scaling factor
    m = np.sqrt( (1-freqs)/freqs/n )   # same as sqrt(1/nk - 1/n) for empirical freqs
    for k in range(0,cl_count):
        diff = mu[:,k]-mup  
        cat[:,k] = diff/(m[k] * sc) # t-scores
    was_diagonal = diagonal # was diagonal used or not
    if not diagonal:
        if verbose:
            print("Computing inverse correlation matrix (pooled across classes) product")
        try:
            catdict = crossprod_powcor_shrink(xc, cat, alpha=-0.5, lambda_cor=lambda_cor, 
                                         verbose=False)
            cat = catdict["cp_powr"]
            regularisation["lambda_cor"] = catdict["lambda_cor"] if lambda_cor is None else lambda_cor
            lambda_estimated = True if lambda_cor is None else False
            if verbose:
                if lambda_estimated:  
                    print("Estimating optimal shrinkage intensity lambda (correlation matrix):", 
                          regularisation["lambda_cor"])
                else:
                    print("Specified shrinkage intensity lambda (correlation matrix):", 
                          regularisation["lambda_cor"])
        except np.linalg.LinAlgError:
            was_diagonal = True # this can happen if SVD doesn't converge
    ###
    return dict(regularisation=regularisation, freqs=freqs, cat=cat, was_diagonal=was_diagonal)
