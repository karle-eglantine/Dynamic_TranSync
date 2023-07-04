# This script is written by E. Karlé to conduct the experiments of the article Dynamic Ranking and Translation Synchornization https://arxiv.org/pdf/2207.01455.pdf
#
# This script contains the main function to compute eigenvalues of the penalized Laplacian L(lambda) for different values of lambda.
# The graphs are generated as Erdös Renyi G(n,p(t)) with p(t) chosen as log(n)/n.

import random
import numpy as np
import pickle
import sys
import scipy.stats as ss
import scipy.sparse.linalg
import os

sys.path.append('modules')

import mle_module as mle
import graph_module as graph
import simulation_module as sim
import drc_module as drc
import smoothness_module as smooth

import time
start = time.time()

# Fixed parameters
N = 100 # Number of items
T = 200 # Number of time points

list_lambda = [0,1e-2,1e-1,1,10] # Values of lambda for the computation of L(lambda)

# Parameters of smoothness and connectivity
alpha = 1 # Smoothness is ST = 1/T^alpha
p1 = np.log(N) # Connectivity of graphs is chosen randomly between p1/N and p2/N 
p2 = np.log(N)


def eigs_L_lam(N,T,alpha,p1,p2,list_lam):

    random.seed(0)
    np.random.seed(0)

    # Compute E as it only depends on N and T
    S_T = 1/T**alpha
    E = smooth.penalty_E(N,T+1)
    
    # Generate input graphs
    A = sim.get_list_union_connected_gnp(N,T,np.log(N),np.log(N))
    
    # Compute Laplacian matrix 
    Q = graph.diag_incidence(A)
    Lv = Q@Q.T

    eigs = np.zeros((len(list_lambda),N*(T+1)))

    # Computation of spectrum for different values of lambda
    for k,c_lam in enumerate(list_lambda):

        lam = c_lam*(T/S_T)**(2/3)
        L = (Lv+lam*E.T@E).todense()
        eigs[k,:] = np.linalg.eigh(L)[0]

    return eigs