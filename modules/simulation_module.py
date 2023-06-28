import scipy as sc
from scipy import stats
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys

sys.path.append('modules')

import graph_module as graph

##### Simulation ground truth data with Gaussian Process ####

def w_gaussian_process(N, T, mu_parameters, cov_parameters, mu_type = 'constant', cov_type = 'toeplitz'):
    '''
    generate w of shape TxN via a Gaussian process : w = exp(beta) where beta is generated via a Gaussian process
    '''
    if mu_type == 'constant':
        loc, scale = mu_parameters
        mu_start = stats.norm.rvs(loc = loc,scale = scale,size = N,random_state = 100)
        mu = [np.ones(T) * mu_start[i] for i in range(N)]
    if cov_type == 'toeplitz':
        alpha, r = cov_parameters
    ##### strong auto-correlation case, off diagonal  = 1 - T^(-alpha) * |i - j|^r
    off_diag = 1 - np.float_power(T,-alpha) * np.float_power(np.arange(1,T + 1),r)
    cov_single_path = sc.linalg.toeplitz(off_diag,off_diag)

    return np.exp(np.array([np.random.multivariate_normal(mean = mu[i],cov = cov_single_path,size = 1).ravel() for i in range(N)]).T)

def z_gaussian_process(N, T, mu_parameters, cov_parameters, mu_type = 'constant', cov_type = 'toeplitz'):
    '''
    generate z of shape TxN via a Gaussian process
    '''
    if mu_type == 'constant':
        loc, scale = mu_parameters
        mu_start = stats.norm.rvs(loc = loc,scale = scale,size = N,random_state = 100)
        mu = [np.ones(T) * mu_start[i] for i in range(N)]
    if cov_type == 'toeplitz':
        alpha, r = cov_parameters
    ##### strong auto-correlation case, off diagonal  = 1 - T^(-alpha) * |i - j|^r
    off_diag = 1 - np.float_power(T,-alpha) * np.float_power(np.arange(1,T + 1),r)
    cov_single_path = sc.linalg.toeplitz(off_diag,off_diag)

    return np.array([np.random.multivariate_normal(mean = mu[i],cov = cov_single_path,size = 1).ravel() for i in range(N)]).T

# or generate,N functions of time, indep of T
def generate_w(N,T):
    '''return a TxN vector with w_i(t) = 2+2*(i-1)/n + cos(2*pi*t)'''
    w = np.zeros((T,N))
    for i in range(N):
        for t in range(T):
            w[t,i] = 2+2*i/np.sqrt(N) + np.cos(2*np.pi*t/T)
    return w

#### Simulation of observed graphs #### 

def get_list_union_connected_gnp(N,T,c1 = 1,c2 = 10):
    '''
    get a list of T+1 adjacency matrices of G(n,p) graphs on the input grid such that union graph is connected
    We forbid that a graph is completely empty.
    -------------
    Output :
    A: (T+1)-N-N array where A[t,:,:] is the adjacency matrix of a G(n,p(t)) at time t. p(t) is drawned uniformly from [c_1/n,c_2/n]
    '''
    A = np.zeros((T+1,N,N))
    union = graph.union_graph(A,T,0.5) # Union of all the graphs
    while not(graph.connected(union)):
        for t in range(T+1):
            while np.all(A[t,:,:]==0):
                p = np.random.uniform(c1/N,c2/N)
                for i in range(N):
                    for j in range(i+1,N):
                        A[t,i,j] = np.random.binomial(1,p)
                        A[t,j,i] = A[t,i,j]
        union = graph.union_graph(A,T,0.5)
    return A

def get_list_all_connected_gnp(N,T,p):
    '''
    get a list of T+1 adjacency matrices of connected G(n,p) graphs on the input grid
    -------------
    Output :
    A: (T+1)-N-N array where A[t,:,:] is the adjacency matrix of a G(n,p(t)) at time t. p(t) is drawned uniformly from [c_1/n,c_2/n]
    '''
    A = np.zeros((T+1,N,N))
    for t in range(T+1):
        while not(graph.connected(A[t,:,:])):
            #p = np.random.uniform(c1/N,c2/N)
            for i in range(N):
                for j in range(i+1,N):
                    A[t,i,j] = np.random.binomial(1,p)
                    A[t,j,i] = A[t,i,j]
    return A

def get_one_connected_gnp(N,c1 = 1,c2 = 10):
    '''
    get an adjacency matrice of a connected G(n,p) graph, p(t) is drawned uniformly from [c_1/n,c_2/n].
    Give
    -------------
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
    get (T+1)*L comparison matrices
    -------------
    Output :
Yl: (T+1)-L-N-N array where Yl[t,l,:,:] is a matrix of pairwise information y_ij^l(t), following Bernoulli distribution of parameter p_ij(t) = w_tj/(w_ti+w_tj) where w = exp(z)
    '''
    Yl = np.zeros((T+1,L,N,N))
    w = np.exp(z)
    for t in range(T+1):
        for l in range(L):
            for i in range(N):
                for j in range(i+1,N):
                    Yl[t,l,i,j] = np.random.binomial(1,w[t,j]/(w[t,j]+w[t,i]))
                    Yl[t,l,j,i] = 1-Yl[t,l,i,j]
    return Yl

def get_comparison_data_transync(z):
    '''
    z : (T+1)-N vector of true strength
    A : (T+1)-N-N array of T+1 adjacency matrices
    ----------
    Output : 
    Y : (T+1)*N array where the blocks for each time t are stacked.
    Y_{ij}(t) = z[t,i]-z[t,j] + gaussian noise.
    '''
    T,N = np.shape(z)
    Y = np.zeros((T,N,N))
    for t in range(T):
        for i in range(N):
            for j in range(1,N):
                eps = np.random.normal(0,1)
                Y[t,i,j] = z[t,i]-z[t,j] + eps
    
    return np.array(Y)



def get_comparison_data_transync_vector(z,A,signal_noise = False):
    '''
    z : (T+1)-N vector of true strength
    A : (T+1)-N-N array of T+1 adjacency matrices
    ----------
    Output : 
    Y : (T+1)*N array where the blocks for each time t are stacked.
    Y_{ij}(t) = z[t,i]-z[t,j] + gaussian noise.
    '''
    T,N = np.shape(A)[:2]
    Y = []
    if signal_noise:
        Noise = []
        Signal = []
    for t in range(T):
        edges = graph.list_edges(A[t])
        for i in range(len(edges)):
            a,b = edges[i]
            eps = np.random.normal(0,1)
            Y.append(z[t,a]-z[t,b] + eps)
            if signal_noise:
                Noise.append(eps)
                Signal.append(z[t,a]-z[t,b])
            
    if signal_noise:
        return np.array(Y),np.array(Noise),np.array(Signal)
    else:
        return np.array(Y)



print('simu')