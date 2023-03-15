# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 10:51:56 2022

@author: camer
"""
from itertools import product
from numpy import random, array, stack

def matrixFactorization(M, N, R, P, Q, K, steps, alpha, beta):
    '''Given a test matrix R with dimensions MxN, and the initial decomposed
    prediction matrix P dim. MxK and Q dim. KxN, compares P.Q to R and adjusts 
    using the factors alpha and beta over the given number of steps to converge
    error between the non-zero elements of R and P.Q to zero. Zero elements 
    are replaced with predictions.'''
    
    for s in range(steps):
        # iterate over the rows of R s number of times
        for m,n in product(range(M),range(N)):   
                # if the current element is non-zero, see similarity to the training matrix
            if R[m][n] != 0:
                # get error between R and P.Q
                e = R[m][n] - P[m,:]@Q[:,n]
                    # adjust P and Q and predict unrated items
                for k in range(K):
                    P[m][k] = P[m][k] + alpha*(2*e*Q[k][n] - beta*P[m][k])
                    Q[k][n] = Q[k][n] + alpha*(2*e*P[m][k] - beta*Q[k][n])
                        
    return P, Q, P@Q
    
if __name__ == "__main__":
    
    R=array([[5,3,0,1], 
              [4,0,0,1], 
              [1,1,0,5], 
              [1,0,0,4], 
              [0,1,5,4]])
    
    # R=array([[4,3,0,1,2], 
    #           [5,0,0,1,0], 
    #           [1,2,1,5,4], 
    #           [1,0,0,4,0], 
    #           [0,1,5,4,0],
    #           [5,5,0,0,1]])

    M, N, K = len(R), len(R[0]), 2
    
    P = stack([random.ranf(K) for m in range(M)], axis=0)
    Q = stack([random.ranf(N) for k in range(K)], axis=0)
    
    steps = 5000
    alpha = 0.0002
    beta = 0.02
    
    nP, nQ, nR = matrixFactorization(M, N, R, P, Q, K, steps, alpha, beta)
    
    