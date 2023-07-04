# This script is written by E. Karlé to conduct the experiments of the article Dynamic Ranking and Translation Synchornization https://arxiv.org/pdf/2207.01455.pdf
#
# This script contains the main function to perform analysis on synthetic data generated according to the Dynamic BTL model.
# The user can choose amongst the Dynamic Least Squares method, the Dynamic Projection method the Dynamic Rank Centrality method and the Maximum Likelihood method.
# This function outputs the mean MSE across Monte Carlo runs for each estimator and each value of T.

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
B = 40 # Number of Monte Carlo runs
list_T = [10,30,50,70,100,130,150,170,200] # Number of time points

# Parameters of smoothness and connectivity
alpha = 1 # Smoothness is ST = 1/T^alpha
p1 = 1 # Connectivity of graphs is chosen randomly between p1/N and p2/N 
p2 = np.log(N)

L = 10 # Each observation is a mean of L Bernoulli in the Dynamic BTL model
c_delta = 0.5 # constant used to construct the neighborhood in the DRC method

# Estimation method to use
dls_flag = True # Dynamic Least Squares method
dproj_flag = True # Dynamic Projection method
drc_flag = True # Dynamic Rank Centrality method
mle_flag = True # Maximum Likelihood method

def btl(N,list_T,B,alpha,p1,p2,L,c_delta,dls_flag,dproj_flag,drc_flag,mle_flag):

    random.seed(0)
    np.random.seed(0)

    # Initialization
    MSE_dls = np.zeros(len(list_T))
    MSE_dproj = np.zeros(len(list_T))
    MSE_drc = np.zeros(len(list_T))
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
        if dls_flag:
            z_dls = np.zeros((N_out,B,N))
            MSE_dls_T = np.zeros((N_out,B))
        if dproj_flag:
            z_dproj = np.zeros((N_out,B,N))
            MSE_dproj_T = np.zeros((N_out,B))

        if drc_flag:
            z_drc = np.zeros((N_out,B,N))
            MSE_drc_T = np.zeros((N_out,B))

        if mle_flag:
            z_mle = np.zeros((N_out,B,N))
            MSE_mle_T = np.zeros((N_out,B))

        # Smoothness 
        ST = T**(-alpha)
        E = smooth.penalty_E(N,T+1) # smoothness operator
        eigs_E,V_E = smooth.eigs_E(N,T) # eigenpairs

        # Hyper parameter tau_z for construction of z* 
        tau_z = (np.pi*ST/(np.sqrt(N-1)*T))**(2/3)# Optimal tau for construction of z*
        V_z = V_E[:,eigs_E <= tau_z] # Select only eigenvectors with eigenvalues less than tau_z

        # Hyper parameter tau_dproj for Dynamic Projection method
        if dproj_flag:
            tau_dproj = (ST/T)**(2/3)# Optimal tau for ST smoothness in projection algorithm
            V_dproj = V_E[:,eigs_E <= tau_dproj]

        # V_z used to create z^* and V_dproj is used in the projection method

        # Bootstrap loop
        for b in range(B):
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
            R = tools.obs_btl(Y,A,L) # Restrain to observed graphs. Vector of length sum_{t=0...T}(# E_t). No zeros in Y_vector
            # Y_vector is used in Dynamic Least Squares and in Dynamic Projection method
            # Y is used in DRC and in MLE methods

            ########## Estimations and MSE for different models ##########

            # Compute useful matrices for both DLS and DProj method 
            Q = graph.diag_incidence(A)
            Lv = Q@Q.T

            ### DLS ###
            if dls_flag:
                # Optimal lambda for smoothness ST
                lam = (T/ST)**(2/3)
                z_hat_dls = scipy.sparse.linalg.lsqr(Lv+lam*E.T@E,Q@np.log(R))[0] # minres is least square solver adapted for symmetric matrices
                z_dls[:,b,:] = z_hat_dls.reshape((N_out,N),order='F')

                # Center the estimator
                for i in range(N_out):
                    z_dls[i,b,:] = z_dls[i,b,:] - np.mean(z_dls[i,b,:])

                    # MSE
                    MSE_dls_T[i,b] = np.linalg.norm(z_dls[i,b,:]-z_T[i,b,:])**2 # L2 norm square of error for centered vectors z^* and z_hat

            ### DProj ###
            if dproj_flag:
                # Unconstrained Least Square
                z_hat = scipy.sparse.linalg.lsqr(Lv,Q@np.log(R))[0]

                # Projection 
                z_dproj[:,b,:] = (V_dproj@V_dproj.T@z_hat).reshape((N_out,N),order='F')

                # Center the estimator
                for i in range(N_out):
                    z_dproj[i,b,:] = z_dproj[i,b,:] - np.mean(z_dproj[i,b,:])

                    # MSE
                    MSE_dproj_T[i,b] = np.linalg.norm(z_dproj[i,b,:]-z_T[i,b,:])**2 # L2 norm square of error for centered vectors z^* and z_hat

            ### DRC  ###
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


            ### MLE  ###
            if mle_flag:
                data_MLE = np.transpose(Y,(0,2,1))

                # Estimation
                h = T**(-3/4)
                T_list = np.arange(0,1+1/step_out)
                ks_data = mle.kernel_smooth(data_MLE,h,T_list)
                z_mle[:,b,:] = mle.gd_bt(data = ks_data,verbose=False)[1]

                # Center estimate
                for i in range(N_out):
                    z_mle[i,b,:] = z_mle[i,b,:] - np.mean(z_mle[i,b,:])

                # MSE
                MSE_mle_T[:,b] = [np.linalg.norm(z_mle[i,b,:]-z_T[i,b,:]) for i in range(N_out)]

        # Mean of all the MSE and Smoothness over bootstraps experiments

        if dls_flag:
            MSE_dls[k] = np.mean(MSE_dls_T)

        if dproj_flag:
            MSE_dproj[k] = np.mean(MSE_dproj_T)

        if drc_flag:
            MSE_drc[k] = np.mean(MSE_drc_T)

        if mle_flag:
            MSE_mle[k] = np.mean(MSE_mle_T)        

    return [MSE_dls,MSE_dproj,MSE_drc,MSE_mle]
