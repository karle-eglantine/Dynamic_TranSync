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
import ls_module as ls

import time
start = time.time()

# Fixed parameters
N = 100
T = 200
list_lambda = [0,1e-2,1e-1,1,10]
alpha = 1 # Smoothness is ST = 1/T^alpha
p1 = np.log(N)
p2 = np.log(N)


def eigs_L_lam(N,T,alpha,p1,p2,list_lam):

    random.seed(0)
    np.random.seed(0)

    # Compute E as it only depends on N and T
    S_T = 1/T**alpha
    E = ls.penalty_E(N,T+1)
    
    # Generate input graphs
    A = sim.get_list_union_connected_gnp(N,T,np.log(N),np.log(N))
    
    # Compute Laplacian matrix 
    Q = graph.diag_incidence(A)
    Lv = Q@Q.T

    eigs = np.zeros((len(list_lambda),N*(T+1)))

    # Computation of spectrum for different values of lambda
    for k,c_lam in enumerate(list_lambda):
        print(c_lam)

        lam = c_lam*(T/S_T)**(2/3)
        L = (Lv+lam*E.T@E).todense()
        eigs[k,:] = np.linalg.eigh(L)[0]

    return eigs

eigs = eigs_L_lam(N,T,1,p1,p2,list_lambda)

with open('eigs.txt','wb') as e:
    pickle.dump(eigs,e)