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

# Fixed parameters
N = 100
B = 40
list_T = [10,30,50,70,100,130,150,170,200]

# Parameters of smoothness and connectivity
alpha = 1 # Smoothness is ST = 1/T^alpha
p1 = 1 # Connectivity of graphs is chosen randomly between p1/N and p2/N 
p2 = np.log(N)

L = 10
c_delta = 0.5

# Estimation method to use
ls_flag = True
proj_flag = True
drc_flag = True
mle_flag = True


random.seed(0)
np.random.seed(0)

# Initialization
if ls_flag:
    MSE_ls = np.zeros(len(list_T))

if proj_flag:
    MSE_proj = np.zeros(len(list_T))
    
if drc_flag:
    MSE_drc = np.zeros(len(list_T))

if mle_flag:
    MSE_mle = np.zeros(len(list_T))

for k,T in enumerate(list_T): # adapt final arrays containing the results
    # Output grid
    step_out = 1/T
    output_grid = np.arange(0,1+step_out/2,step_out) # Grid: 0,1/T,2/T,...,1
    N_out = len(output_grid)
    grid = np.arange(0,T+1) # Grid: 0,1,2,...,T

    #### Initialization of result arrays ####
    # Ground_truth
    z_T = np.zeros((N_out,B,N))
    
    # Estimation methods
    if ls_flag:
        z_ls = np.zeros((N_out,B,N))
        MSE_ls_T = np.zeros((N_out,B))
    if proj_flag:
        z_proj = np.zeros((N_out,B,N))
        MSE_proj_T = np.zeros((N_out,B))
    
    if drc_flag:
        z_drc = np.zeros((N_out,B,N))
        MSE_drc_T = np.zeros((N_out,B))
    
    if mle_flag:
        z_mle = np.zeros((N_out,B,N))
        MSE_mle_T = np.zeros((N_out,B))
    
    # Smoothness
    ST = T**(-alpha)
    
    # Eigenpairs of E.T@E
    if os.path.exists("eigenpairs_E/eigenvectors_E_n"+str(N)+"_T"+str(T)+".txt"):
        # Load eigenpairs of E
        with open("eigenpairs_E/eigenvectors_E_n"+str(N)+"_T"+str(T)+".txt", "rb") as v:
            V_E = pickle.load(v)

        with open("eigenpairs_E/eigenvalues_E_n"+str(N)+"_T"+str(T)+".txt", "rb") as e:
            eigs_E = pickle.load(e)
    else:
        # Compute eigenvectors and eigenvalues of E and save it in eigenpairs_E folder
        eigs_E,V_E = eigs.eigs_E(N,T)
    
    # Hyper parameter tau for construction of z* 
    tau_z = (np.pi*ST/(np.sqrt(N-1)*T))**(2/3)# Optimal tau for construction of z*
    V_z = V_E[:,eigs_E <= tau_z] # Select only eigenvectors with eigenvalues less than tau_z
    
    # Hyper parameter tau for Projection method
    if proj_flag:
        tau_proj = (ST/T)**(2/3)# Optimal tau for ST smoothness in projection algorithm
        V_proj = V_E[:,eigs_E <= tau_proj]
        
    # V_z used to create z^* and V_proj is used in the projection method
    
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

        
        ########## Generate input graphs ##########
        
        A = sim.get_list_union_connected_gnp(N,T,p1,p2)

        ########## Generate pairwise information from TranSync Model ##########
        
        Yl = sim.get_comparison_data_BTL(N,T,L,z_T[:,b,:])
        Y = A*np.mean(Yl,axis=1) # Put zeros where no edges were observed. Array of size (T+1-N-N)
        R = ls.obs_btl(Y,A,L) # Restrain to observed graphs. Vector of length sum_{t=0...T}(# E_t). No zeros in Y_vector
        # Y_vector is used in Penalized LS and in Projection method
        # Y is used in DRC and in MLE methods

        ########## Estimations and MSE for different models ##########
        
        # Compute useful matrices for both LS and Projection method 
        Q = graph.diag_incidence(A)
        Lv = Q@Q.T
        
        ### Penalised LS ###
        if ls_flag:
            # Optimal lambda for smoothness ST
            lam = (T/ST)**(2/3)
            z_hat_ls = scipy.sparse.linalg.lsqr(Lv+lam*E.T@E,Q@np.log(R))[0] # minres is least square solver adapted for symmetric matrices
            z_ls[:,b,:] = z_hat_ls.reshape((N_out,N),order='F')

            # Center the estimator
            for i in range(N_out):
                z_ls[i,b,:] = z_ls[i,b,:] - np.mean(z_ls[i,b,:])

                # MSE
                MSE_ls_T[i,b] = np.linalg.norm(z_ls[i,b,:]-z_T[i,b,:])**2 # L2 norm square of error for centered vectors z^* and z_hat
        
        ### Projection method ###
        if proj_flag:
            # Unconstrained Least Square
            z_hat = scipy.sparse.linalg.lsqr(Lv,Q@np.log(R))[0]
            
            # Projection 
            z_proj[:,b,:] = (V_proj@V_proj.T@z_hat).reshape((N_out,N),order='F')
            
            # Center the estimator
            for i in range(N_out):
                z_proj[i,b,:] = z_proj[i,b,:] - np.mean(z_proj[i,b,:])

                # MSE
                MSE_proj_T[i,b] = np.linalg.norm(z_proj[i,b,:]-z_T[i,b,:])**2 # L2 norm square of error for centered vectors z^* and z_hat
        
        ### DRC method ###
        if drc_flag:
            # Choose right delta 
            delta = max(1/2,c_delta*T**(2/3))
            vec_delta = delta*np.ones(N_out)
            vec_delta = drc.get_delta_drc(A,vec_delta,output_grid)
            
            # Estimate at each time point on the grid
            for i,t in enumerate(output_grid):
                z_drc[i,b,:] = np.log(drc.RC_dyn(t,Y,A,vec_delta[i],tol = 1e-12))
                
                # Center z_RC
                z_drc[i,b,:] = z_drc[i,b,:] - np.mean(z_drc[i,b,:])
                
                # MSE
                MSE_drc_T[i,b] = np.linalg.norm(z_drc[i,b,:]-z_T[i,b,:])**2
            
            
        ### MLE method ###
        if mle_flag:
            data_MLE = np.transpose(Y,(0,2,1))
            
            # Estimation
            h = T**(-3/4)
            T_list = np.arange(0,1+1/step_out)
            ks_data = mle.kernel_smooth(data_MLE,h,T_list)
            z_mle[:,b,:] = mle.gd_bt(data = ks_data,verbose=True)[1]
            
            # Center estimate
            for i in range(N_out):
                z_mle[i,b,:] = z_mle[i,b,:] - np.mean(z_mle[i,b,:])
                
            # MSE
            MSE_mle_T[:,b] = [np.linalg.norm(z_mle[i,b,:]-z_T[i,b,:]) for i in range(N_out)]

    # Mean of all the MSE and Smoothness over bootstraps experiments
    
    if ls_flag:
        MSE_ls[k] = np.mean(MSE_ls_T)
        
    if proj_flag:
        MSE_proj[k] = np.mean(MSE_proj_T)
        
    if drc_flag:
        MSE_drc[k] = np.mean(MSE_drc_T)
        
    if mle_flag:
        MSE_mle[k] = np.mean(MSE_mle_T)        
####################### Save results ######################

if ls_flag:
    with open("MSE_ls_btl_alpha_"+str(alpha)+".txt", "wb") as mse_ls:
        pickle.dump(MSE_ls, mse_ls)

if proj_flag:
    with open("MSE_proj_btl_alpha_"+str(alpha)+".txt", "wb") as mse_proj:
        pickle.dump(MSE_proj, mse_proj)
        
if drc_flag:
    with open("MSE_drc_btl_alpha_"+str(alpha)+".txt", "wb") as mse_drc:
        pickle.dump(MSE_drc, mse_drc)
        
if mle_flag:
    with open("MSE_mle_btl_alpha_"+str(alpha)+".txt", "wb") as mse_mle:
        pickle.dump(MSE_mle, mse_mle)

