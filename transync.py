# This script is written by E. Karlé to conduct the experiments of the article Dynamic Ranking and Translation Synchornization https://arxiv.org/pdf/2207.01455.pdf
#
# This script contains the main function to perform analysis on synthetic data generated according to the Dynamic TranSync model.
# The user can choose amongst the naive Least Squares method, the Dynamic Least Squares method and the Dynamic Projection method.
# This function outputs the MSE of each the estimators for all Monte Carlo runs and all values of T, and the mean MSE across Monte Carlo runs for each estimator and each value of T.

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
import tools_module as tools
import smoothness_module as smooth

# Fixed parameters
N = 100 # Number of items
B = 20 # Number of Monte Carlo runs
list_T = [10,30,50,70,100,130,150,170,200] # Number of time points

# Parameters of smoothness and connectivity
alpha = 1 # Smoothness is ST = 1/T^alpha
p1 = 1 # Connectivity of graphs is chosen randomly between p1/N and p2/N 
p2 = np.log(N)

# Estimation method to use
ls_flag = True # Naive Least Squares (only make sense when all the graphs are connected)
dls_flag = True # Dynamic Least Squares
dproj_flag = True # Dynamic Projection




def transync_analysis(N,list_T,B,alpha,p1,p2,ls_flag,dls_flag,dproj_flag):
    
    random.seed(0)
    np.random.seed(0)
    
    # Initialization
    data_ls = []
    MSE_ls = np.zeros(len(list_T))

    data_dls = []
    MSE_dls = np.zeros(len(list_T))

    data_dproj = []
    MSE_dproj = np.zeros(len(list_T))

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

        if dproj_flag:
            z_dproj = np.zeros((N_out,B,N))
            MSE_dproj_T = np.zeros((N_out,B))

        # Smoothness 
        ST = T**(-alpha)
        E = smooth.penalty_E(N,T+1) # smoothness operator
        eigs_E,V_E = smooth.eigs_E(N,T) # eigenpairs 

        # Hyper parameter tau_z for construction of z* 
        tau_z = (np.pi*ST/(np.sqrt(N-1)*T))**(2/3)# Optimal tau for construction of z*
        V_z = V_E[:,eigs_E <= tau_z] # Select only eigenvectors with eigenvalues less than tau_z

        # Hyper parameter tau_dproj for Projection method
        if dproj_flag:
            tau_dproj = (ST/T)**(2/3)# Optimal tau for ST smoothness in projection algorithm
            V_dproj = V_E[:,eigs_E <= tau_dproj]

        # V_z used to create z^* and V_dproj is used in the projection method

        

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

            Y = sim.get_comparison_data_transync_vector(z_T[:,b,:],A)

            ########## Estimations and MSE for different models ##########

            # Compute useful matrices for both DLS and DProj method 
            Q = graph.diag_incidence(A)
            Lv = Q@Q.T

            ### DLS ###
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

            ### DProj ###
            if dproj_flag:
                # Unconstrained Least Square
                z_hat = scipy.sparse.linalg.lsqr(Lv,Q@Y)[0]

                # Projection 
                z_dproj[:,b,:] = (V_dproj@V_dproj.T@z_hat).reshape((N_out,N),order='F')

                # Center the estimator
                for i in range(N_out):
                    z_dproj[i,b,:] = z_dproj[i,b,:] - np.mean(z_dproj[i,b,:])

                # MSE
                MSE_dproj_T[:,b] = [np.linalg.norm(z_dproj[i,b,:]-z_T[i,b,:])**2 for i in range(N_out)] # L2 norm square of error for centered vectors z^* and z_hat

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

        if dproj_flag:
            data_dproj.append(np.ravel(MSE_dproj_T)) # List of length len(list_T), containing arrays of size T*B
            # It contains all the MSE values for all bootstrap loop and for all t=0...T
            MSE_dproj[k] = np.mean(MSE_dproj_T)

            
    return [MSE_ls,MSE_dls,MSE_dproj,data_ls,data_dls,data_dproj]