import scipy as sc
from scipy import stats
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys

import graph_module as graph

def borda_count(Y,A,delta):
    T, N = A.shape[:2]
    Y_delta = np.sum(Y,axis=0)
    A_delta = np.sum(A,axis=0)
    borda_count = np.sum(Y_delta, axis=0) / np.sum(A_delta,axis=0)
    return borda_count

def error_metric(w,sigma):
    s = 0
    N = np.shape(w)[0]
    for i in range(N):
        for j in range(i+1,N):
            cond = (w[i]-w[j])*(sigma[i]-sigma[j])>0
            if cond:
                s =+ (w[i]-w[j])**2 # Only count pairs where ranking is incorrect
    return np.sqrt(s/(2*N*np.linalg.norm(w)**2))

def rank_list(v):
    output = [0] * len(v)
    for i, x in enumerate(sorted(range(len(v)), key=lambda y: v[y])):
        output[x] = i
    return np.array(output)

def av_dif_rank(w1,w2):
    # both w should be T-by-N matrices
    T, N = w1.shape
    result = [0] * T
    for t in range(T):
        result[t] = np.mean(abs(rank_list(w1[t]) - rank_list(w2[t])))
    return result

def loocv_borda(Y,A,delta_list,t,num_loocv = 200):
    '''
    t : time at which we want to recover the ranks
    '''    
    T,N = np.shape(A)[:2]
    # Create a pool of pi to choose from
    pis = np.zeros((len(delta_list),N))
    for i,delta in enumerate(delta_list):
        pis[i,:] = sim.borda_count(t,Y,A, delta)
    
    
    indices = np.transpose(np.nonzero(A)) # Array of all (t,i,j) possible combinations to choose from
    N_comp = np.shape(indices)[0] # total number of comparisons
    
    error = np.zeros(len(delta_list))
    for l, delta in enumerate(delta_list):
        error_delta = np.zeros(num_loocv)
        for k in range(num_loocv):
            Y_loocv = Y.copy()
            A_loocv = A.copy()
        
            rand_match = np.random.randint(N_comp) # random number between 0 and Total number of comparisons
            rand_index = indices[rand_match,:] # Select the tuple (t,i,j) corresponding to the rand_match comparison
            s,i,j = tuple(rand_index)
        
            # Remove the test value from the data
            Y_loocv[s,i,j] = max(Y_loocv[s,i,j]-1,0) # if all observations at these time where 0, then Y[t,i,j] stays 0.
            Y_loocv[s,j,i] = max(Y_loocv[s,j,i]-1,0)
            A_loocv[s,i,j] = 0
            A_loocv[s,j,i] = 0  
            
            # Fit model and compute prediction error
            pi = sim.borda_count(s/T,Y_loocv,A_loocv, delta) # vector of length N
            ranks = ss.rankdata(-pi,method='average')
            error_delta[k] = (Y[s,i,j]==1)*(ranks[i]< ranks[j])
            
        error[l] = np.mean(error_delta)
        
    index = max(idx for idx, val in enumerate(error) if val == np.min(error[~np.isnan(error)]))
    delta_star = delta_list[index]
    pi_star = pis[index,:]
    
    return delta_star,pi_star

