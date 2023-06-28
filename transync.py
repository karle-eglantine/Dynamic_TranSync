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
import eigenvectors_E as eigs

# Fixed parameters
N = 100
B = 20
list_T = [10,30,50,70,100,130,150,170,200]

# Parameters of smoothness and connectivity
alpha = 1 # Smoothness is ST = 1/T^alpha
p1 = 1 # Connectivity of graphs is chosen randomly between p1/N and p2/N 
p2 = np.log(N)

# Estimation method to use
ls_flag = True
dls_flag = True
proj_flag = True




def transync_analysis(N,list_T,B,alpha,p1,p2,ls_flag,dls_flag,proj_flag):
    
    random.seed(0)
    np.random.seed(0)
    
    # Initialization
    data_ls = []
    MSE_ls = np.zeros(len(list_T))

    data_dls = []
    MSE_dls = np.zeros(len(list_T))

    data_proj = []
    MSE_proj = np.zeros(len(list_T))

    for k,T in enumerate(list_T): # adapt final arrays containing the results
        # Output grid
        step_out = 1/T
        output_grid = np.arange(0,1+step_out/2,step_out) # Grid: 0,1/T,2/T,...,1
        N_out = len(output_grid)
        grid = np.arange(0,T+1) # Grid: 0,1,2,...,T

        #### Initialization of result arrays ####
        # Ground_truth
        z_T = np.zeros((N_out,B,N))
        smoothness_T = np.zeros(B)

        # Estimation methods
        if ls_flag:
            z_ls = np.zeros((N_out,B,N))
            MSE_ls_T = np.zeros((N_out,B))

        if dls_flag:
            z_dls = np.zeros((N_out,B,N))
            MSE_dls_T = np.zeros((N_out,B))

        if proj_flag:
            z_proj = np.zeros((N_out,B,N))
            MSE_proj_T = np.zeros((N_out,B))

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
            if dls_flag:
                # Optimal lambda for smoothness ST
                lam = (T/ST)**(2/3)
                z_hat_dls = scipy.sparse.linalg.lsqr(Lv+lam*E.T@E,Q@Y)[0]
                z_dls[:,b,:] = z_hat_dls.reshape((N_out,N),order='F')

                # Center the estimator
                for i in range(N_out):
                    z_dls[i,b,:] = z_dls[i,b,:] - np.mean(z_dls[i,b,:])

                # MSE
                MSE_dls_T[:,b] = [np.linalg.norm(z_dls[i,b,:]-z_T[i,b,:])**2 for i in range(N_out)] # L2 norm square of error for centered vectors z^* and z_hat

            ### Projection method ###
            if proj_flag:
                # Unconstrained Least Square
                z_hat = scipy.sparse.linalg.lsqr(Lv,Q@Y)[0]

                # Projection 
                z_proj[:,b,:] = (V_proj@V_proj.T@z_hat).reshape((N_out,N),order='F')

                # Center the estimator
                for i in range(N_out):
                    z_proj[i,b,:] = z_proj[i,b,:] - np.mean(z_proj[i,b,:])

                # MSE
                MSE_proj_T[:,b] = [np.linalg.norm(z_proj[i,b,:]-z_T[i,b,:])**2 for i in range(N_out)] # L2 norm square of error for centered vectors z^* and z_hat

            ### Naive LS ###
            if ls_flag:
                # Unconstrained Least Square
                z_ls[:,b,:] = (scipy.sparse.linalg.lsqr(Lv,Q@Y)[0]).reshape((N_out,N),order='F')

                # Center the estimator
                for i in range(N_out):
                    z_ls[i,b,:] = z_ls[i,b,:] - np.mean(z_ls[i,b,:])

                # MSE
                MSE_ls_T[:,b] = [np.linalg.norm(z_ls[i,b,:]-z_T[i,b,:])**2 for i in range(N_out)] # L2 norm square of error for centered vectors z^* and z_hat

        # Mean signal to noise ration over bootstrap experiments

        if ls_flag:
            data_ls.append(np.ravel(MSE_ls_T)) # Save all errors to be able to construct boxplot
            MSE_ls[k] = np.mean(MSE_ls_T)

        if dls_flag:
            data_dls.append(np.ravel(MSE_dls_T))
            MSE_dls[k] = np.mean(MSE_dls_T)

        if proj_flag:
            data_proj.append(np.ravel(MSE_proj_T)) # List of length len(list_T), containing arrays of size T*B
            # It contains all the MSE values for all bootstrap loop and for all t=0...T
            MSE_proj[k] = np.mean(MSE_proj_T)

            
    return [MSE_ls,MSE_dls,MSE_dproj,data_ls,data_dls,data_proj]