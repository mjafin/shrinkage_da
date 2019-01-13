# -*- coding: utf-8 -*-
"""
Variance Shrinkage

@author Miika Ahdesmaki, Korbinian Strimmer
"""
from __future__ import print_function, division
import numpy as np
from wt_scale import wt_moments
from shrink_misc import minmax
from shrink_intensity import estimate_lambda_var

def pvt_svar(x, lambda_var = None, w = None, verbose = False):
    """Private function estimating variance shrikage 
    
    Non-public function to compute variance shrinkage estimator
    """
    if lambda_var is None:
        lambda_var = estimate_lambda_var(x, w, verbose)
        lambda_var_estimated = True
    else:
        lambda_var = minmax(lambda_var)
        if verbose:
            print("Specified shrinkage intensity lambda.var (variance vector): ", lambda_var)
        lambda_var_estimated = False
    # compute empirical variances
    v = wt_moments(x, w)["var"]
    # compute shrinkage target
    target = np.median(v)
    vs = lambda_var*target + (1-lambda_var)*v
    return vs, lambda_var, lambda_var_estimated