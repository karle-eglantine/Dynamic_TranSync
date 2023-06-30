# This module provide tools to use the Dynamic Rank Centrality algorithm [Karle and Tyagi]  https://arxiv.org/abs/2109.13743
# This code comes from https://github.com/karle-eglantine/Dynamic_Rank_Centrality

import numpy as np
import scipy

import graph_module as graph

# Main function for DRC analysis
def RC_dyn(t,Y,A,delta,tol = 1e-12):
    '''
    Get the normalized estimator pihat_RC(t)
    ------------
    Input :
        Y: (T+1)-N-N array containig the pairwise comparison information, averaged on the L workers at each time
        A: (T+1)-N-N array, A[t,:,:] is the adjacency matrix of data at time t
        delta: parameter of neighborhood size
        tol: tolerance to approximate the eigenvalues equal to 1. 
    Output :
        Estimator pi_RC, array of length N
    '''
    T,N = np.shape(A)[:2]
    N_delta = graph.neighborhood(t,delta,T-1)
    A_delta = graph.union_graph(A,delta,t)
    P = transition_matrix(A_delta,Y,N_delta)
    eigval,eigvec = scipy.linalg.eig(P,left=True,right=False)
    pi_RC = eigvec[:,abs(eigval - 1) < tol][:,0]
    return pi_RC/sum(pi_RC)

# Computation of neighborhood parameter delta 
def get_delta_drc(A,vec_delta,output_grid):
    '''
    Generate list of T+1 adjacency matrices and vector of values of delta such that for any t in the output_grid, G_delta(t) is connected
    Input :
        A : T-N-N array such that A[t,:,:] is the adjacency of graph at time t. We ensured that union of all the graphs are connected
        vec_delta : vector of same length as output_grid, initialized to be the constant vector with value delta^* = c_delta*T**(2/3)/(N*L**1/3)
        output_grid : list of times at which we want to recover the ranks
    Output:
        vec_delta : updated values of delta, for wich G_delta(t) is connected (t in output_grid)
    '''
    T,N = np.shape(A)[:2]
    grid = np.arange(0,1+1/(2*T),1/T)
    for i,t in enumerate(output_grid):
        A_delta = graph.union_graph(A,vec_delta[i],t)
        N_delta = graph.neighborhood(t,vec_delta[i],T)
        while sum(N_delta) == 0 or not graph.connected(A_delta):
            # Add graphs on each side of the neighborhood
            idx = update_neighborhood(N_delta) # List of indexes of points of the grid in the neighborhood. We add (if possible) one grid point on the left and on the right to the old neighborhood.
            vec_delta[i] = T*max(t-grid[idx[0]],grid[idx[-1]]-t)
            A_delta = graph.union_graph(A,vec_delta[i],t)
    return vec_delta

# Tools for DRC

# Neighborhood construction
def update_neighborhood(neigh):
    '''
    Extend a neighborhood of one grid unit on the left and on the right (if possible) on a grid of size T
    Input:
        neigh: array of boolean of length T, neigh[t] = True if t is a timepoint in the neighborhood
    Output:
        updated array of booleans
    '''
    T = np.shape(neigh)[0]
    idx = np.nonzero(neigh)[0]
    if np.shape(idx)[0] == T: # The neighborhood is already equal to the complete grid
        return idx
    else:
        new_neigh = neigh
        if idx[0] == 0: # Can't complete the neighborhood on the left : already include time t = 0
            new_neigh[idx[-1]+1] = True # Add one point on the right
        else:
            new_neigh[idx[0]-1] = True # Add a point on the left
            if idx[-1]!= T-1: # Can also add a point on the right
                new_neigh[idx[-1]+1] = True
    return np.nonzero(new_neigh)[0]


# Construction of tranition matrix
def transition_matrix(A_delta,Y,N_delta):
    '''
    Compute the transition matrix of the union graph at time t
    Input : 
        A_delta : adajcency matrix of the union graph
        Y : (T+1)-N-N array containig the pairwise comparison information, averaged on the L workers at each time
        N_delta : neighborhood
    Output : N-N array
    '''
    N = np.shape(Y)[1]
    transition_matrix = np.zeros((N,N))
    d = 2*N*graph.graph_proba(A_delta)
    for i in range(N):
        for j in range(N):
            if (i != j) & (A_delta[i,j] != 0):
                transition_matrix[i,j] = sum(Y[N_delta,i,j])/(d*A_delta[i,j])
        transition_matrix[i,i] = 1-np.sum(transition_matrix[i,:])
    return transition_matrix



