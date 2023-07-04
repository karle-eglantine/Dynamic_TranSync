# This script is written by E. Karlé to conduct the experiments of the article Dynamic Ranking and Translation Synchornization https://arxiv.org/pdf/2207.01455.pdf
#
# This module computes the smoothness operator E = M^t kron C^t and the eigenpairs of EE^t (and save them)
# where MM^t = laplacian of the path graph on T+1 vertices
#   and CC^t = laplacian of the complete graph on N vertices


import random
import numpy as np
import pickle
import sys
import scipy.stats as ss
import scipy.sparse.linalg
import os

sys.path.append('modules')

import graph_module as graph

def penalty_E(n,T):
    '''T = number of input graphs'''
    C = graph.incidence_Kn(n)
    M = graph.incidence_path(T)
    return scipy.sparse.kron(M.T,C.T)
        
def eigs_E(N,T):
    ''' Compute (or load) the eigenpairs of E^T E = MM^T kron CC^T and save them into the eigenpairs_E folder.'''
    
    if not os.path.exists("eigenpairs_E"):
        os.makedirs("eigenpairs_E")
    
    # If eigenpairs already computed, load them
    if os.path.exists("eigenpairs_E/eigenvectors_E_n"+str(N)+"_T"+str(T)+".txt"):
        # Load eigenpairs of E
        with open("eigenpairs_E/eigenvectors_E_n"+str(N)+"_T"+str(T)+".txt", "rb") as v:
            V_E = pickle.load(v)

        with open("eigenpairs_E/eigenvalues_E_n"+str(N)+"_T"+str(T)+".txt", "rb") as e:
            eigs_E = pickle.load(e)
    # Computation of the eigenpairs        
    else:

        E = penalty_E(N,T+1) # Smoothness operator
        E = E.todense()
        eigs_E,V_E = scipy.linalg.eigh(E.T@E) # Compute the eigenpairs

        # Save them in the eigenpairs_E folder
        with open("eigenpairs_E/eigenvectors_E_n"+str(N)+"_T"+str(T)+".txt", "wb") as v:
            pickle.dump(V_E, v)
        with open("eigenpairs_E/eigenvalues_E_n"+str(N)+"_T"+str(T)+".txt", "wb") as e:
            pickle.dump(eigs_E, e)

    return eigs_E,V_E
        