# -*- coding: utf-8 -*-
"""
Shrinkage discriminant analysis (prediction)

@author Miika Ahdesmaki, Korbinian Strimmer
"""
from __future__ import print_function, division
import numpy as np
from sys import exit

def predict_sda(sda_object, Xtest, verbose = False):
    """SDA feature ranking
    
    Parameters
    ----------
    sda_object : dict
        dictionary from sda containing model parameters
    Xtest : numpy array
        samples-in-rows matrix. Number of columns must match the number of 
        variables used in training of the provided sda_object
    
    Returns
    -------
    dict
        dictionary containin class predictions and posterior probabilities. 
        Predicted class is the one with highest posterior probability.
    """
    n, p = Xtest.shape
    alpha = sda_object["alpha"]
    beta = sda_object["beta"]
    #cl_count = len(alpha)
    if p != beta.shape[1]:
        raise ValueError("Different number of predictors in sda object (" + str(beta.shape[1]) + ") and in Xtest (" + str(p) + ")")
    if verbose:
        print("Prediction uses ",p," features")
    probs = (np.matmul(beta, Xtest.T) + alpha).T
    probs = np.exp(probs - np.max(probs, axis=1, keepdims=True))
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    
    #yhat = sda_object["groups"][np.argmax(probs, axis=1)]
    yhat = np.array([sda_object["groups"][myind] for myind in np.argmax(probs, axis=1)])
    
    return dict(predicted_class = yhat, posterior = probs)