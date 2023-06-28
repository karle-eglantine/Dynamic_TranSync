import random
import numpy as np
import pickle
import sys
import scipy.stats as ss
import scipy.sparse.linalg
import os

sys.path.append('modules')

import ls_module as ls
        
def eigs_E(N,T):
    ''' eigenvectors of E^T E = MM^T kron CC^T'''
    
    E = ls.penalty_E(N,T+1)
    E = E.todense()
    
    eigs_E,V_E = scipy.linalg.eigh(E.T@E)
    
    if not os.path.exists("eigenpairs_E"):
        os.makedirs("eigenpairs_E")
        
    with open("eigenpairs_E/eigenvectors_E_n"+str(N)+"_T"+str(T)+".txt", "wb") as v:
        pickle.dump(V_E, v)
    with open("eigenpairs_E/eigenvalues_E_n"+str(N)+"_T"+str(T)+".txt", "wb") as e:
        pickle.dump(eigs_E, e)
    
    return eigs_E,V_E
        