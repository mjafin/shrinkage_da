# -*- coding: utf-8 -*-
"""
Shrinkage discriminant analysis (feature ranking)

@author Miika Ahdesmaki, Korbinian Strimmer
"""
from __future__ import print_function, division
import numpy as np
from sys import exit
from catscore import catscore

def sda_ranking(Xtrain, L, lambda_cor = None, lambda_var = None, lambda_freqs = None, ranking_score = "entropy", diagonal=False, verbose=False):
    """SDA feature ranking
    
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
    ranking_score : string
        One of "entropy", "avg" or "max". For two class classification the choices
        converge to the same result, thus only important for multi class classification
    diagonal : bool
        If True, skip correlation adjustment and assume diagonal model (False)
    verbose : bool
        Verbose mode (False).
    
    Returns
    -------
    dictionary
        Dictionary containing order of ranked features, summarised cat scores, 
        cat-scores, regularisation parameters and prior frequencies
    """
    if ranking_score not in ["entropy","avg","max"]:
        exit("ranking_score must be one of 'entropy', 'avg' or 'max'")
    cat = catscore(Xtrain, L, lambda_cor=lambda_cor, lambda_var=lambda_var, 
                   lambda_freqs=lambda_freqs, diagonal=diagonal, verbose=verbose)
    cl_count = cat["cat"].shape[1]
    if ranking_score == "entropy":
        score = np.matmul(np.power(cat["cat"], 2), 1-cat["freqs"])  # weighted sum of squared CAT scores
    if ranking_score == "avg":
        score = np.sum(np.power(cat["cat"], 2), axis=1) / cl_count # average of squared CAT-scores
    if ranking_score == "max":
        score = np.max(np.power(cat["cat"], 2), axis=1)          # max of squared CAT-scores
    idx = np.argsort(score)[::-1] # decreasing sort order of cat scores
    
    # Future implementation of FDR evaluation goes here
    
    # Without FDR just return sort order, scores, cats
    
    return dict(idx=idx, score = score[idx], cat = cat["cat"][idx,:], 
                regularisation = cat["regularisation"], freqs = cat["freqs"], was_diagonal = cat["was_diagonal"])