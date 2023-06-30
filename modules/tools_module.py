# This script is written by E. Karlé to conduct the experiments of the article Dynamic Ranking and Translation Synchornization https://arxiv.org/pdf/2207.01455.pdf
#
# This module contains function used to reshape data to fit the studied models 

import scipy
import numpy as np
import sys

import graph_module as graph


def obs_transync(Y,A):
    '''
    Create the vector R of all the observations, of length = sum_{t=0}^T #edges in graph G(t)
    R[t,i,j] = Y_ij(t)
    '''
    T= np.shape(A)[0]
    R = []
    for t in range(T):
        edges = graph.list_edges(A[t,:,:])
        for i in range(len(edges)):
            a,b = edges[i]
            R.append(Y[t,a,b])
    return R

def obs_btl(Y,A,L):
    '''
    Create the vector R of all the observations, of length = sum_{t=0}^T #edges in graph G(t)
    R[t,i,j] = Y_ij(t)/Y_ji(t)
    '''
    T= np.shape(A)[0]
    R = []
    for t in range(T):
        edges = graph.list_edges(A[t,:,:])
        for i in range(len(edges)):
            a,b = edges[i]
            if Y[t,a,b] == 0:
                R.append(1/(2*L-1))
            elif Y[t,a,b] == 1:
                R.append(2*L-1)
            else:
                R.append(Y[t,a,b]/Y[t,b,a])
    return R

