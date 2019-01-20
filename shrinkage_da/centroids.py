# -*- coding: utf-8 -*-
"""
Group centroids and (pooled) variances

@author Miika Ahdesmaki, Korbinian Strimmer
"""
from __future__ import print_function, division
import numpy as np
from sys import exit
from corpcor.shrink_misc import minmax
from corpcor.shrink_estimates import var_shrink

def centroids(x, L, lambda_var = None, lambda_freqs = None, var_groups=False, centered_data=False, verbose=False):
    """Estimate centroids for the Bayes classifier (SDA)
    
    Parameters
    ----------
    x : array
        Samples-in-rows matrix.
    L : vector array
        Class label vector/array. Must match number of rows in x.
    lambda_var : float or list
        Shrinkage parameter for variances or list specified_lambda_var = lambda_varif separate ones used per class
    lambda_freqs : float
        Shrinkage parameter for class prevalences
    var_groups : bool
        Estimate separate variances for each class/group (False)
    centered_data : bool
        Data pre-centred (False)
    verbose : bool
        Verbose mode (False)
    
    Returns
    -------
    dictionary
        Dictionary containing information about samples, class frequencies, 
        means, variances and centred data
    """
    n, p = x.shape
    if len(L) != n:
        exit("Number of rows in input matrix x must match the number of class labels")
    samples, cl_count, cl_names, idx = pvt_groups(L)
    # do some checking for lambda_var
    if lambda_var is None:
        auto_shrink = True
    else:
        if not isinstance(lambda_var, list) and (isinstance(lambda_var, float) or isinstance(lambda_var, int)):
            lambda_var = [lambda_var] # make into list
        auto_shrink = False
        if var_groups:
            if len(lambda_var) == 1:
                specified_lambda_var = lambda_var[0] * np.ones(cl_count + 1)
            else:
                if len(lambda_var) != (cl_count+1):
                    exit("lambda_var of variance shrinkage intensities must be single value or list of values with one intensity for each group plus one value for the pooled variance.")
                specified_lambda_var = lambda_var
        else:
            specified_lambda_var = [lambda_var[0]]
        # restrict lambda between 0 and 1
        specified_lambda_var = [minmax(xtemp) for xtemp in specified_lambda_var]
    if verbose:
        print("Number of variables: ", p)
        print("Number of samples: ", n)
        print("Number of classes: ", cl_count)
    # estimate class frequencies
    lambda_freqs_estimated = True if lambda_freqs is None else False
    freqs, lambda_freqs_est = pvt_freqs_shrink(samples, lambda_freqs=lambda_freqs, verbose=verbose)
    # setup array
    mu = np.zeros((p, cl_count+1)) # means
    xc = np.zeros((n,p))  # storage for centered data
    my_group_lambdas = np.zeros(1)
    if var_groups:
        v = np.zeros((p, cl_count+1)) # storage for variances
        my_group_lambdas = np.zeros((cl_count+1))
    else:
        v = np.zeros((p, 1)) # store only pooled variances
    # compute means and variance in each group
    mu_pooled = np.zeros(p)
    for k in range(0,cl_count):
        my_idx = idx[:, k]
        Xk = x[my_idx, :]
        mu[:, k] = np.average(Xk, axis=0)
        mu_pooled = mu_pooled + freqs[k] * mu[:, k]
        
        xc[my_idx, :] = Xk - mu[:, k] # sweep means
        if var_groups:
            if verbose:
                print("Estimating variances (class #", k, ")")
            if auto_shrink:
                vs, lambda_var_temp, _ = var_shrink(Xk, verbose = verbose)
            else:
                vs,_,_ = var_shrink(Xk, lambda_var = specified_lambda_var[k], verbose=verbose)
            v[:,k] = vs
            my_group_lambdas[k] = lambda_var_temp
    mu[:, cl_count] = mu_pooled
    
    # compute variance
    if verbose:
        print("Estimating variances (pooled across classes")
    if var_groups:
        if auto_shrink:
            v_pool,my_lambda_var,_ = var_shrink(xc, verbose=verbose)
        else:
            v_pool,my_lambda_var,_ = var_shrink(xc, lambda_var = specified_lambda_var[cl_count], verbose=verbose)
        v[:,cl_count] = v_pool*(n-1)/(n-cl_count) # correction factor
        my_group_lambdas[cl_count] = my_lambda_var
    else:
        if auto_shrink:
            v_pool, my_lambda_var,_ = var_shrink(xc, verbose=verbose)
        else:
            v_pool, my_lambda_var,_ = var_shrink(xc, lambda_var = specified_lambda_var[0])
        v[:,0] = v_pool*(n-1)/(n-cl_count) # correction factor
        my_group_lambdas[0] = my_lambda_var
    
    if auto_shrink:
        lambda_var_estimated = True
    else:
        lambda_var_estimated = False
    if not centered_data:
        xc = None
        
    cl_names.append("(pooled)")
    return dict(samples=samples, freqs=freqs, means=mu, variances=v, 
                centered_data=xc, lambda_var_estimated=lambda_var_estimated, 
                var_lambdas=my_group_lambdas,
                freqs_lambda=lambda_freqs_est, freqs_lambda_estimated=lambda_freqs_estimated,
                groups = cl_names)
                
    
def pvt_groups(L):
    L = np.array(L) # make sure it's a numpy array that can be compared
    cl_names = list(set(L)) # unique class labels
    cl_count = len(cl_names)
    idx = np.full((len(L), cl_count), False) # array of False values
    samples = np.zeros(cl_count)
    for k in range(0,cl_count):
        idx[:,k] = (L == cl_names[k])
        samples[k] = np.sum(idx[:,k])
    return samples, cl_count, cl_names, idx

def pvt_freqs_shrink(y, lambda_freqs = None, verbose = False):
    """Estimates the bin frequencies from the counts ‘y’
     using a James-Stein-type shrinkage estimator, where the shrinkage
     target is the uniform distribution.
     """
    target = 1/len(y) # uniform target 
    n = np.sum(y)
    u = y/n
    if lambda_freqs is None:
        if n==1 or n==0:
            lambda_freqs = 1
        else:
            lambda_freqs = pvt_get_lambda_shrink(n, u, target, verbose)
    else:
        if verbose:
            print("Specified shrinkage intensity lambda.freq (frequencies): ", lambda_freqs)
    u_shrink = lambda_freqs * target + (1 - lambda_freqs) * u
    return u_shrink, lambda_freqs

def pvt_get_lambda_shrink(n, u, t, verbose):
    # *unbiased* estimator of variance of u
    varu = u*(1-u)/(n-1)
    # misspecification
    msp = sum( np.power(u-t, 2) )
    # estimate shrinkage intensity
    my_lambda = 1 if msp == 0 else np.sum( varu ) / msp
    my_lambda = minmax(my_lambda) # [0,1]
    if verbose:
        print("Estimating optimal shrinkage intensity lambda.freq (frequencies): ", my_lambda)
    return my_lambda
    
