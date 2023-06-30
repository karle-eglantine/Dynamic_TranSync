# This script is written by E. Karlé to conduct the experiments of the article Dynamic Ranking and Translation Synchornization https://arxiv.org/pdf/2207.01455.pdf
#
# This module implements the cross validation procedure for both the DLS and DProj methods, for two performance criteria : the number of upsets and the MSE

import scipy
import numpy as np
import sys

import graph_module as graph
import tools_module as tools

### DLS Method ###

# Cross validation with Number of Upsets criterion
def cv_dls_up(Y,A,E,lambda_list,num_loocv = 200):
    '''
    Proceeds to cross-validation to find optimal parameter lambda using the number of upsets as a criterion and return the optimal lambda^* as well as the DLS estimator for this value of parameter.
    Input:
        Y: observation matrices
        A: adjacency matrices
        E: smoothness operator
        lambda_list: candidates for the hyperparameter lambda
        num_loocv : numbre of cross-validation iterations
    Output:
        lam_up : optimal hyperparameter 
        z_up : estimator obtained by DLS method for the choice lambda = lam_up
    '''
    
    N_out,N = np.shape(A)[:2]
    indices = []
    N_comp = np.zeros(N_out)

    for k in range(N_out):
        indices.append(np.transpose(np.nonzero(A[k,:,:]))) 
        # Each element of the list is the array of all (i,j) possible combinations to choose from at time k
        N_comp[k] = np.shape(indices[k])[0] # total number of comparisons at time k
    
    error_up = np.zeros(len(lambda_list))
    for l,lam in enumerate(lambda_list):
        error_up_lam = np.zeros(num_loocv)
        for b in range(num_loocv):
            Y_loocv = Y.copy()
            A_loocv = A.copy()
            
            test_set = np.zeros((N_out,2),dtype=int)
            
            # We remove one observation at each timepoint
            for k in range(N_out):
                rand_match = np.random.randint(N_comp[k]) # random number between 0 and Total number of comparisons at time k
                rand_index = indices[k][rand_match,:] # Select the tuple (i,j) corresponding to the rand_match comparison at time k
                i,j = tuple(rand_index)
        
                # Remove the test value from the data
                Y_loocv[k,i,j] = 0
                Y_loocv[k,j,i] = 0
                A_loocv[k,i,j] = 0
                A_loocv[k,j,i] = 0 
                
                test_set[k,:] = [i,j]
            
            # Fit model and compute prediction error
            Y_vector = tools.obs_transync(Y_loocv,A_loocv)
            Q = graph.diag_incidence(A_loocv)
            Lv = Q@Q.T
            z = scipy.sparse.linalg.lsqr(Lv+lam*E.T@E,Q@Y_vector)[0]
            z = z.reshape((N_out,N),order='F')
            
            # Center estimate and compare the estimate of strengths difference and the observed strengths difference for the items we remove
            
            e_up = np.zeros(N_out)
            for k in range(N_out):
                # Center
                z[k,:] = z[k,:] - np.mean(z[k,:])
                # Compute estimation error on test values
                i,j = test_set[k,:]
                e_up[k] = (np.sign(Y[k,i,j]) != np.sign((z[k,i]-z[k,j])))
                # e_up[k] : Boolean indicating if estimation at time k recover the right ordering between i and j
                                
            error_up_lam[b] = np.mean(e_up)
                                
        # Mean error over bootsrap experiments
        error_up[l] = np.mean(error_up_lam)
    
    # Select lambda that minimizes the Upsets error
    index_up = max(idx for idx, val in enumerate(error_up) if val == np.min(error_up[~np.isnan(error_up)]))
    lam_up = lambda_list[index_up]
                                
    # Compute corresponding estimate z
    Y_vec = tools.obs_transync(Y,A)
    Q = graph.diag_incidence(A)
    Lv = Q@Q.T
    
    z_up = scipy.sparse.linalg.lsqr(Lv+lam_up*E.T@E,Q@Y_vec)[0]
    z_up = z_up.reshape((N_out,N),order='F')
    
    # Center the estimate
    for k in range(N_out):
        z_up[k,:] = z_up[k,:]-np.mean(z_up[k,:])
    
    return lam_up,z_up


def cv_dls_mse(Y,A,E,lambda_list,num_loocv = 200):
    '''
    Proceeds to cross-validation to find optimal parameter lambda using the MSE as a criterion and return the optimal lambda^* as well as the DLS estimator
    Input:
        Y: observation matrices
        A: adjacency matrices
        E: smoothness operator
        lambda_list: candidates for the hyperparameter lambda
        num_loocv : numbre of cross-validation iterations
    Output:
        lam_mse : optimal hyperparameter 
        z_mse : estimator obtained by DLS method for the choice lambda = lam_mse
    '''
    N_out,N = np.shape(A)[:2]
    indices = []
    N_comp = np.zeros(N_out)

    for k in range(N_out):
        indices.append(np.transpose(np.nonzero(A[k,:,:]))) 
        # Each element of the list is the array of all (i,j) possible combinations to choose from at time k
        N_comp[k] = np.shape(indices[k])[0] # total number of comparisons at time k
    
    error_mse = np.zeros(len(lambda_list))
    for l,lam in enumerate(lambda_list):
        error_mse_lam = np.zeros(num_loocv)
        for b in range(num_loocv):
            Y_loocv = Y.copy()
            A_loocv = A.copy()
            
            test_set = np.zeros((N_out,2),dtype=int)
            
            # We remove one observation at each timepoint
            for k in range(N_out):
                rand_match = np.random.randint(N_comp[k]) # random number between 0 and Total number of comparisons at time k
                rand_index = indices[k][rand_match,:] # Select the tuple (i,j) corresponding to the rand_match comparison at time k
                i,j = tuple(rand_index)
        
                # Remove the test value from the data
                Y_loocv[k,i,j] = 0
                Y_loocv[k,j,i] = 0
                A_loocv[k,i,j] = 0
                A_loocv[k,j,i] = 0 
                
                test_set[k,:] = [i,j]
            
            # Fit model and compute prediction error
            Y_vector = tools.obs_transync(Y_loocv,A_loocv)
            Q = graph.diag_incidence(A_loocv)
            Lv = Q@Q.T
            z = scipy.sparse.linalg.lsqr(Lv+lam*E.T@E,Q@Y_vector)[0]
            z = z.reshape((N_out,N),order='F')
            
            # Center estimate and compare the estimate of strengths difference and the observed strengths difference for the items we remove
            
            e_mse = np.zeros(N_out)
            for k in range(N_out):
                # Center
                z[k,:] = z[k,:] - np.mean(z[k,:])
                # Compute estimation error on test values
                i,j = test_set[k,:]
                e_mse[k] = (Y[k,i,j]-(z[k,i]-z[k,j]))**2
                # e_up[k] : Boolean indicating if estimation at time k recover the right ordering between i and j
                                
            error_mse_lam[b] = np.mean(e_mse)
                                
        # Mean error over bootsrap experiments
        error_mse[l] = np.mean(error_mse_lam)
    
    # Select lambda that minimizes the MSE error
    index_mse = max(idx for idx, val in enumerate(error_mse) if val == np.min(error_mse[~np.isnan(error_mse)]))
    lam_mse = lambda_list[index_mse]
                                
    # Compute corresponding estimate z
    Y_vec = tools.obs_transync(Y,A)
    Q = graph.diag_incidence(A)
    Lv = Q@Q.T
    z_mse = scipy.sparse.linalg.lsqr(Lv+lam_mse*E.T@E,Q@Y_vec)[0]
    z_mse = z_mse.reshape((N_out,N),order='F')
    
    # Center the estimate
    for k in range(N_out):
        z_mse[k,:] = z_mse[k,:]-np.mean(z_mse[k,:])
    
    return lam_mse,z_mse

### Dynmiac Projection method ###

def cv_dproj_up(Y,A,V_E,eigs_E,tau_list,num_loocv = 200):
    '''
    Proceeds to cross-validation to find optimal parameter tau using the number of upsets as a criterion and return the optimal tau^* as well as the DProj estimator
    Input:
        Y: observation matrices
        A: adjacency matrices
        V_E: eigenvectors of the smoothness operator
        eigs_E: eigenvalues of the smoothness operator
        tau_list: candidates for the hyperparameter tau
        num_loocv : numbre of cross-validation iterations
    Output:
        tau_up : optimal hyperparameter 
        z_up : estimator obtained by DLS method for the choice tau = tau_up
    '''
    N_out,N = np.shape(A)[:2]
    T = N_out-1
    indices = []
    N_comp = np.zeros(N_out)

    for k in range(N_out):
        indices.append(np.transpose(np.nonzero(A[k,:,:]))) 
        # Each element of the list is the array of all (i,j) possible combinations to choose from at time k
        N_comp[k] = np.shape(indices[k])[0] # total number of comparisons at time k
    
    error_up = np.zeros(len(tau_list))
    for l,tau in enumerate(tau_list):
        error_up_tau = np.zeros(num_loocv)
        
        # Compute projection matrix for this value of tau
        V_tau = V_E[:,eigs_E <= tau]
        
        for b in range(num_loocv):
            Y_loocv = Y.copy()
            A_loocv = A.copy()
            
            test_set = np.zeros((N_out,2),dtype=int)
            
            # We remove one observation at each timepoint
            for k in range(N_out):
                rand_match = np.random.randint(N_comp[k]) # random number between 0 and Total number of comparisons at time k
                rand_index = indices[k][rand_match,:] # Select the tuple (i,j) corresponding to the rand_match comparison at time k
                i,j = tuple(rand_index)
        
                # Remove the test value from the data
                Y_loocv[k,i,j] = 0
                Y_loocv[k,j,i] = 0
                A_loocv[k,i,j] = 0
                A_loocv[k,j,i] = 0 
                
                test_set[k,:] = [i,j]
            
            # Fit model and compute prediction error
            Y_vector = tools.obs_transync(Y_loocv,A_loocv)
            Q = graph.diag_incidence(A_loocv)
            Lv = Q@Q.T 
            
            z = scipy.sparse.linalg.lsqr(Lv,Q@Y_vector)[0]
            z_proj = (V_tau@V_tau.T@z).reshape((N_out,N),order='F')
            z = z.reshape((N_out,N),order='F')
            # Compare the estimate of strengths difference and the observed strengths difference for the items we remove
            e_up = np.zeros(N_out)
            for k in range(N_out):
                # Center
                z_proj[k,:] = z_proj[k,:] - np.mean(z_proj[k,:])
                # Compare to test value
                i,j = test_set[k,:]
                e_up[k] = (np.sign(Y[k,i,j]) != np.sign((z[k,i]-z[k,j]))) 
                # e_up[k] : Boolean indicating if estimation at time k recover the right ordering between i and j
                                
            error_up_tau[b] = np.mean(e_up)
                                
        # Mean error over bootsrap experiments
        error_up[l] = np.mean(error_up_tau)

    
    # Select tau that minimizes the Upsets
    index_up = max(idx for idx, val in enumerate(error_up) if val == np.min(error_up[~np.isnan(error_up)]))
    tau_up = tau_list[index_up]
                                
    # Compute corresponding estimate z
    Y_vec = tools.obs_transync(Y,A)
    Q = graph.diag_incidence(A)
    Lv = Q@Q.T
    z = (scipy.sparse.linalg.lsqr(Lv,Q@Y_vec)[0])
    
    # Projection
    # Compute projection matrix for this value of tau
    V_up = V_E[:,eigs_E <= tau_up]
    # V_tau is used in the projection method
    z_up = (V_up@V_up.T@z).reshape((N_out,N),order='F')
    
    # Center
    for k in range(N_out):
        z_up[k,:] = z_up[k,:] - np.mean(z_up[k,:])
    
    return tau_up,z_up

def cv_dproj_mse(Y,A,V_E,eigs_E,tau_list,num_loocv = 200):
    '''
    Proceeds to cross-validation to find optimal parameter tau using the MSE as a criterion and return the optimal tau^* as well as the DProj estimator
    Input:
        Y: observation matrices
        A: adjacency matrices
        V_E: eigenvectors of the smoothness operator
        eigs_E: eigenvalues of the smoothness operator
        tau_list: candidates for the hyperparameter tau
        num_loocv : numbre of cross-validation iterations
    Output:
        tau_mse : optimal hyperparameter 
        z_mse : estimator obtained by DLS method for the choice tau = tau_mse
    '''
    N_out,N = np.shape(A)[:2]
    T = N_out-1
    indices = []
    N_comp = np.zeros(N_out)

    for k in range(N_out):
        indices.append(np.transpose(np.nonzero(A[k,:,:]))) 
        # Each element of the list is the array of all (i,j) possible combinations to choose from at time k
        N_comp[k] = np.shape(indices[k])[0] # total number of comparisons at time k
    
    error_mse = np.zeros(len(tau_list))
    for l,tau in enumerate(tau_list):
        error_mse_tau = np.zeros(num_loocv)
        
        # Compute projection matrix for this value of tau
        V_tau = V_E[:,eigs_E <= tau]
        
        for b in range(num_loocv):
            Y_loocv = Y.copy()
            A_loocv = A.copy()
            
            test_set = np.zeros((N_out,2),dtype=int)
            
            # We remove one observation at each timepoint
            for k in range(N_out):
                rand_match = np.random.randint(N_comp[k]) # random number between 0 and Total number of comparisons at time k
                rand_index = indices[k][rand_match,:] # Select the tuple (i,j) corresponding to the rand_match comparison at time k
                i,j = tuple(rand_index)
        
                # Remove the test value from the data
                Y_loocv[k,i,j] = 0
                Y_loocv[k,j,i] = 0
                A_loocv[k,i,j] = 0
                A_loocv[k,j,i] = 0 
                
                test_set[k,:] = [i,j]
            
            # Fit model and compute prediction error
            Y_vector = tools.obs_transync(Y_loocv,A_loocv)
            Q = graph.diag_incidence(A_loocv)
            Lv = Q@Q.T 
            
            z = scipy.sparse.linalg.lsqr(Lv,Q@Y_vector)[0]
            z_proj = (V_tau@V_tau.T@z).reshape((N_out,N),order='F')
            z = z.reshape((N_out,N),order='F')
            # Compare the estimate of strengths difference and the observed strengths difference for the items we remove
            e_mse = np.zeros(N_out)
            for k in range(N_out):
                # Center
                z_proj[k,:] = z_proj[k,:] - np.mean(z_proj[k,:])
                # Compare to test value
                i,j = test_set[k,:]
                e_mse[k] = (Y[k,i,j]-(z[k,i]-z[k,j]))**2
                                
            error_mse_tau[b] = np.mean(e_mse)
                                
        # Mean error over bootsrap experiments
        error_mse[l] = np.mean(error_mse_tau)
    
    # Select tau that minimizes the MSE
    index_mse = max(idx for idx, val in enumerate(error_mse) if val == np.min(error_mse[~np.isnan(error_mse)]))
    tau_mse = tau_list[index_mse]
                                
    # Compute corresponding estimate z
    Y_vec = tools.obs_transync(Y,A)
    Q = graph.diag_incidence(A)
    Lv = Q@Q.T
    z = (scipy.sparse.linalg.lsqr(Lv,Q@Y_vec)[0])
    
    # Projection
    # Compute projection matrix for this value of tau
    V_mse = V_E[:,eigs_E <= tau_mse]
    # V_tau is used in the projection method
            
    z_mse = (V_mse@V_mse.T@z).reshape((N_out,N),order='F')
    
    # Center
    for k in range(N_out):
        z_mse[k,:] = z_mse[k,:] - np.mean(z_mse[k,:])
    
    return tau_mse,z_mse