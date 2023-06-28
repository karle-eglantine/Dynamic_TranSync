import random
import numpy as np
import pickle
import sys
import scipy.stats as ss
import scipy.sparse.linalg

sys.path.append('modules')

import mle_module as mle
import graph_module as graph
import simulation_module as sim
import drc_module as drc
import ls_module as ls

import time
start = time.time()

# Fixed parameters
N = 50
T = 100
B = 20
list_lambda = [1e-2,1e-1,1,10]
alpha = 1 # Smoothness is ST = 1/T^alpha
S_T = T**alpha
random.seed(0)
np.random.seed(0)

    
# Compute E as it only depends on N and T
E = ls.penalty_E(N,T+1)

eigs = np.zeros((len(list_lambda),N*(T+1)))
    
# Different values of lambda
for k,c_lam in enumerate(list_lambda):
    print(c_lam)
    
    ########## Generate input graphs ##########

    A = sim.get_list_union_connected_gnp(N,T,np.log(N),np.log(N))

    # Compute Laplacian matrix 
    Q = graph.diag_incidence(A)
    Lv = Q@Q.T
    print(np.shape(Lv))

    lam = c_lam*(T/S_T)**(2/3)
    L = (Lv+lam*E.T@E).todense()
    eigs[k,:] = np.linalg.eigh(L)[0]
        
end = time.time()
print('Execution time :',end-start)

####################### Save results ######################
with open("eigs_L_lam_ST_"+str(alpha)+"_p_"+str(p)+".txt",'wb')as eig:
    pickle.dump(eigs,eig)