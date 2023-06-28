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
import eigenvectors_E as eigs

import time
start = time.time()

# Fixed parameters
N = 100
T = 200
B = 10

# Grid of lambda for performance estimation
list_lambda = [10**i for i in range(-1,7)]

# Parameters of smoothness and connectivity
alpha = 1 # Smoothness is ST = 1/T^alpha
p1 = np.log(N) # Connectivity of graphs is chosen randomly between p1/N and p2/N 
p2 = np.log(N)

# Smoothness
ST = T**(-alpha)


random.seed(0)
np.random.seed(0)

# Initialization
data_dls = []
MSE_dls = np.zeros(len(list_lambda))
    
# Output grid
step_out = 1/T
output_grid = np.arange(0,1+step_out/2,step_out) # Grid: 0,1/T,2/T,...,1
N_out = len(output_grid)
grid = np.arange(0,T+1) # Grid: 0,1,2,...,T

# Load eigenpairs of E.T@E
if os.path.exists("eigenpairs_E/eigenvectors_E_n"+str(N)+"_T"+str(T)+".txt"):
        # Load eigenpairs of E
        with open("eigenpairs_E/eigenvectors_E_n"+str(N)+"_T"+str(T)+".txt", "rb") as v:
            V_E = pickle.load(v)

        with open("eigenpairs_E/eigenvalues_E_n"+str(N)+"_T"+str(T)+".txt", "rb") as e:
            eigs_E = pickle.load(e)
    else:
        # Compute eigenvectors and eigenvalues of E and save it in eigenpairs_E folder
        eigs_E,V_E = eigs.eigs_E(N,T)ues_E_N"+str(N)+"_T"+str(T)+".txt", "rb") as e:
    eigs_E = pickle.load(e)

tau_z = (np.pi*ST/(np.sqrt(N-1)*T))**(2/3)# Optimal tau for construction of z*
V_z = V_E[:,eigs_E <= tau_z] # Select only eigenvectors with eigenvalues less than tau_z
# V_z used to create z^*
    

for k,lam in enumerate(list_lambda): # adapt final arrays containing the results
    
    #### Initialization of result arrays ####
    # Ground_truth
    z_T = np.zeros((N_out,B,N))
    smoothness_T = np.zeros(B)
    
    # Estimation methods
    z_dls = np.zeros((N_out,B,N))
    MSE_dls_T = np.zeros((N_out,B))
        
    
    
    # Compute E as it only depends on N and T
    E = ls.penalty_E(N,T+1)
    
    # Bootstrap loop
    for b in range(B):
        print((T,b))
        # Compute z*
        x = np.random.normal(size = N*N_out)
        z = V_z@V_z.T@(x/np.linalg.norm(x))
        z_T[:,b,:] = z.reshape((N_out,N),order='F')
        
        # Center z^* on each block
        for i in range(N_out):
            z_T[i,b,:] = z_T[i,b,:]-np.mean(z_T[i,b,:])
            
        # Multiply by N in order for the signal to noise ration to be reasonnable
        # In general for all T, the mean ratio seems to be around 10^-2 = 1/N.
        z_T[:,b,:] = N*z_T[:,b,:]
        
        ########## Generate input graphs ##########
        
        A = sim.get_list_union_connected_gnp(N,T,p1,p2)

        ########## Generate pairwise information from TranSync Model ##########
        
        Y = sim.get_comparison_data_transync_vector(z_T[:,b,:],A,False)
        
        ########## Estimations and MSE for different models ##########
        
        # Compute useful matrices for both LS and Projection method 
        Q = graph.diag_incidence(A)
        Lv = Q@Q.T
        
        ### Penalised LS ###
        z_hat_dls = scipy.sparse.linalg.lsqr(Lv+lam*E.T@E,Q@Y)[0]
        z_dls[:,b,:] = z_hat_dls.reshape((N_out,N),order='F')

        # Center the estimator
        for i in range(N_out):
            z_dls[i,b,:] = z_dls[i,b,:] - np.mean(z_dls[i,b,:])

        # MSE
        MSE_dls_T[:,b] = [np.linalg.norm(z_dls[i,b,:]-z_T[i,b,:])**2 for i in range(N_out)] # L2 norm square of error for centered vectors z^* and z_hat

    data_dls.append(np.ravel(MSE_dls_T))
    MSE_dls[k] = np.mean(MSE_dls_T)
        
end = time.time()
print('Execution time :',end-start)

####################### Save results ######################

with open("performance_MSE_dls_transync_alpha_ST_"+str(alpha)+".txt", "wb") as mse_dls:
    pickle.dump(MSE_dls, mse_dls)
with open("performance_data_dls_transync_alpha_ST_"+str(alpha)+".txt", "wb") as Data_dls:
    pickle.dump(data_dls, Data_dls)
