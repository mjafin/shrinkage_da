# -*- coding: utf-8 -*-
"""
Efficient Computation of the Singular Value Decomposition

@author: Miika Ahdesmaki, Korbinian Strimmer
"""
from __future__ import print_function, division
import numpy as np

def positive_svd(m, tol):
    """svd that retains only positive singular values 
    """
    u, d, v = np.linalg.svd(m)
    # determine rank of B  (= rank of m)
    if tol is None: 
      tol = max(m.shape) * max(d) * np.finfo(float).eps 
    Positive = d > tol
    sub_range = np.array(range(0,len(Positive)))[Positive] # subset of indices
    return (d[Positive], u[:, sub_range], v.T[:, sub_range])

def nsmall_svd(m, tol):
    B = np.matmul(m, m.T) # n by n matrix
    u, d, _ = np.linalg.svd(B) # ...whose svd is easy   
    # determine rank of B  (= rank of m)
    if tol is None: 
      tol = B.shape[0] * max(d) * np.finfo(float).eps 
    Positive = d > tol                            
           
    # positive singular values of m  
    d = np.sqrt(d[Positive])
      
    # corresponding orthogonal basis vectors
    u = u[:, Positive]
    v = np.matmul(np.matmul(m.T, u), np.diag(1/d) )   
  
    return (d, u, v)


def psmall_svd(m, tol):
    B = np.matmul(m.T, m) # p by p matrix
    _, d, v = np.linalg.svd(B) # ...whose svd is easy   
    # determine rank of B  (= rank of m)
    if tol is None: 
      tol = B.shape[0] * max(d) * np.finfo(float).eps 
    Positive = d > tol             
           
    # positive singular values of m  
    d = np.sqrt(d[Positive])
      
    # corresponding orthogonal basis vectors (v is different from the v returned by R by a transpose)
    v = v.T[:, Positive]
    u = np.matmul(np.matmul(m, v), np.diag(1/d) )   
    return (d, u, v)


# public functions

# fast computation of svd(m)

# note that the signs of the columns vectors in u and v
# mayfast computation of svd(m) if n >> p be different from that given by svd()

# note that also only positive singular values are returned

def fast_svd(m, tol = None):
    """Fast computation of svd(m)
Note that the signs of the columns vectors in u and v
may be different from that given by svd()

Note that also only positive singular values are returned
    
    Parameters
    ----------
    m : array
        Matrix whose svd is sought.
    tol : float
        Singularity tolerance.
    
    Returns
    -------
    tuple
        Returns (d, u, v) tuple
        
    """
    n, p = m.shape
    EDGE_RATIO = 2 # use standard SVD if matrix almost square
    if n > EDGE_RATIO*p:
        return psmall_svd(m, tol)
    elif EDGE_RATIO*n < p:
        return nsmall_svd(m, tol)
    else: # if p and n are approximately the same
        return positive_svd(m, tol)
