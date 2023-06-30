# This script is written by E. Karlé to conduct the experiments of the article Dynamic Ranking and Translation Synchornization https://arxiv.org/pdf/2207.01455.pdf
#
# This module contains the function used to simulate synthetic data according to the BTL model, 
# as well as a list of G(n,p) graphs satisfying certain connectivity conditions.

import scipy as sc
from scipy import stats
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys

sys.path.append('modules')

import graph_module as graph

#### Simulation of observed graphs #### 

def get_list_union_connected_gnp(N,T,c1 = 1,c2 = 10):
    '''
    Create T+1 adjacency matrices of G(n,p) graphs on the input grid such that union of all the graphs is connected
    We forbid that a graph is completely empty.
    Input:
        N: number of vertices
        T: grid of T+1 points
        c1,c2: for each graph G(t), p(t) is drawned uniformly from [c_1/n,c_2/n]
    Output :
        A: (T+1)-N-N array where A[t,:,:] is the adjacency matrix of a G(n,p(t)) at time t.
    '''
    A = np.zeros((T+1,N,N))
    union = graph.union_graph(A,T,0.5) # Union of all the graphs
    while not(graph.connected(union)): # Restart the process as ling as the union is disconnected
        for t in range(T+1):
            while np.all(A[t,:,:]==0):
                p = np.random.uniform(c1/N,c2/N)
                for i in range(N):
                    for j in range(i+1,N):
                        A[t,i,j] = np.random.binomial(1,p) # edge (i,j) exists with probability p
                        A[t,j,i] = A[t,i,j]
        union = graph.union_graph(A,T,0.5)
    
    return A

def get_list_all_connected_gnp(N,T,p):
    '''
    Create T+1 adjacency matrices of connected G(n,p) graphs on the input grid
    Input:
        N: number of vertices
        T: grid of T+1 points
        p: probability of existence of an edge
    Output :
        A: (T+1)-N-N array where A[t,:,:] is the adjacency matrix of a G(n,p) at time t.
    '''
    A = np.zeros((T+1,N,N))
    for t in range(T+1):
        while not(graph.connected(A[t,:,:])): # Restart if graph at time t is disconnected
            for i in range(N):
                for j in range(i+1,N):
                    A[t,i,j] = np.random.binomial(1,p)# edge (i,j) exists with probability p
                    A[t,j,i] = A[t,i,j]
    return A

def get_one_connected_gnp(N,c1 = 1,c2 = 10):
    '''
    Create an adjacency matrice of a connected G(n,p) graph, p is drawned uniformly from [c_1/n,c_2/n].
    Input:
        N: number of vertices
        c1,c2: p is drawned uniformly from [c_1/n,c_2/n]
    Output :
        A: N-N array adjacency matrix of a connected Gnp.
    '''
    A = np.zeros((N,N))
    while not(graph.connected(A)):
        p = np.random.uniform(c1/N,c2/N)
        for i in range(N):
            for j in range(i+1,N):
                A[i,j] = np.random.binomial(1,p)
                A[j,i] = A[i,j]
    return A

#### Simulation of pairwise observations from true weights z or w = exp(z) ####

def get_comparison_data_BTL(N,T,L,z):
    '''
    Create (T+1)*L comparison matrices according to the BTL model with weights w = exp(z)
    Input:
        N: number of vertices
        T: grid of size T+1
        L: for each edge, we observe L Bernoulli variables with parameter p_ij(t) = w_tj/(w_ti+w_tj)
        z: true weights
    Output:
        Yl: (T+1)-L-N-N array where Yl[t,l,:,:] is a matrix of containing the Bernoulli pairwise information y_ij^l(t)
    '''
    Yl = np.zeros((T+1,L,N,N))
    w = np.exp(z) # weights for the BTL model
    for t in range(T+1):
        for l in range(L):
            for i in range(N):
                for j in range(i+1,N):
                    Yl[t,l,i,j] = np.random.binomial(1,w[t,j]/(w[t,j]+w[t,i]))
                    Yl[t,l,j,i] = 1-Yl[t,l,i,j]
    return Yl


def get_comparison_data_transync_vector(z,A):
    '''
    Create a long vector containing all tge observations generated according to the Dynamic TranSync Model with weights z.
    Input:
        z : (T+1)-N vector of true strengths
        A : (T+1)-N-N array of T+1 adjacency matrices
    Output : 
        Y : (T+1)*N array where the blocks for each time t are stacked.
        Y_{ij}(t) = z[t,i]-z[t,j] + gaussian noise.
    '''
    T,N = np.shape(A)[:2]
    Y = []
    for t in range(T):
        edges = graph.list_edges(A[t]) # List of observed edges at time t
        for i in range(len(edges)):
            a,b = edges[i]
            eps = np.random.normal(0,1) # Gaussian noise
            Y.append(z[t,a]-z[t,b] + eps) # Stack all the noisy observations in a long vector Y
        
    return np.array(Y)



print('simu')