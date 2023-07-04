# This script is written by E. Karlé to conduct the experiments of the article Dynamic Ranking and Translation Synchornization https://arxiv.org/pdf/2207.01455.pdf
#
# This script contains the main function used to compute the performance of the Dynamic Projection method for different values of parameter tau. 
# In this function, data are generated according to the Dynamic TranSync model for chosen values of N and T, and the results are averaged over Monte Carlo runs.

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

import time
start = time.time()

# Fixed parameters
N = 100 # Number of items
B = 20 # Number of Monte Carlo runs
T = 200 # Number of time points

# Grid of tau for performance estimation
list_tau = [1e-4,1e-3,1e-2,1e-1,1]

# Parameters of smoothness and connectivity
alpha = 1 # Smoothness is ST = 1/T^alpha
p1 = np.log(N) # Connectivity of graphs is chosen randomly between p1/N and p2/N 
p2 = np.log(N)


def perf_dproj(N,T,B,alpha,p1,p2,list_tau):
    random.seed(0)
    np.random.seed(0)

    # Initialization
    data_dproj = []
    MSE_dproj = np.zeros(len(list_tau))

    # Output grid
    step_out = 1/T
    output_grid = np.arange(0,1+step_out/2,step_out) # Grid: 0,1/T,2/T,...,1
    N_out = len(output_grid)
    grid = np.arange(0,T+1) # Grid: 0,1,2,...,T

    # Smoothness 
    ST = T**(-alpha)
    eigs_E,V_E = smooth.eigs_E(N,T) # eigenpairs
    
    # Hyper parameter tau_z for construction of z* 
    tau_z = (np.pi*ST/(np.sqrt(N-1)*T))**(2/3)# Optimal tau for construction of z*
    V_z = V_E[:,eigs_E <= tau_z] # Select only eigenvectors with eigenvalues less than tau_z

    for k,tau in enumerate(list_tau): 

        #### Initialization of result arrays ####
        # Ground_truth
        z_T = np.zeros((N_out,B,N))
        smoothness_T = np.zeros(B)

        # Estimation methods
        z_dproj = np.zeros((N_out,B,N))
        MSE_dproj_T = np.zeros((N_out,B))


        # Matrix in Dynamic Projection algorithm
        V_dproj = V_E[:,eigs_E <= tau]

        # Bootstrap loop
        for b in range(B):
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

            # Compute useful matrices for both LS and Projection method 
            Q = graph.diag_incidence(A)
            Lv = Q@Q.T

            ### Dynamic Projection method ###
            # Unconstrained Least Square
            z_hat = scipy.sparse.linalg.lsqr(Lv,Q@Y)[0]

            # Projection 
            z_dproj[:,b,:] = (V_dproj@V_dproj.T@z_hat).reshape((N_out,N),order='F')

            # Center the estimator
            for i in range(N_out):
                z_dproj[i,b,:] = z_dproj[i,b,:] - np.mean(z_dproj[i,b,:])

            # MSE
            MSE_dproj_T[:,b] = [np.linalg.norm(z_dproj[i,b,:]-z_T[i,b,:])**2 for i in range(N_out)] # L2 norm square of error for centered vectors z^* and z_hat

        data_dproj.append(np.ravel(MSE_dproj_T)) # List of length len(list_T), containing arrays of size T*B
        # It contains all the MSE values for all bootstrap loop and for all t=0...T
        MSE_dproj[k] = np.mean(MSE_dproj_T)

    return [MSE_dproj,data_dproj]