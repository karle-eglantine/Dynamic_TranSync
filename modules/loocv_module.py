import scipy
import numpy as np
import sys

import graph_module as graph
import ls_module as ls

### LS Method ###
def cv_dls_transync(Y,A,E,lambda_list,num_loocv = 200):
    
    N_out,N = np.shape(A)[:2]
    indices = []
    N_comp = np.zeros(N_out)

    for k in range(N_out):
        indices.append(np.transpose(np.nonzero(A[k,:,:]))) 
        # Each element of the list is the array of all (i,j) possible combinations to choose from at time k
        N_comp[k] = np.shape(indices[k])[0] # total number of comparisons at time k
    
    error_mse = np.zeros(len(lambda_list))
    error_up = np.zeros(len(lambda_list))
    for l,lam in enumerate(lambda_list):
        error_mse_lam = np.zeros(num_loocv)
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
            Y_vector = ls.obs_transync(Y_loocv,A_loocv)
            Q = graph.diag_incidence(A_loocv)
            Lv = Q@Q.T
            z = scipy.sparse.linalg.lsqr(Lv+lam*E.T@E,Q@Y_vector)[0]
            z = z.reshape((N_out,N),order='F')
            
            # Center estimate and compare the estimate of strengths difference and the observed strengths difference for the items we remove
            
            e_mse = np.zeros(N_out)
            e_up = np.zeros(N_out)
            for k in range(N_out):
                # Center
                z[k,:] = z[k,:] - np.mean(z[k,:])
                # Compute estimation error on test values
                i,j = test_set[k,:]
                e_mse[k] = (Y[k,i,j]-(z[k,i]-z[k,j]))**2
                e_up[k] = (np.sign(Y[k,i,j]) != np.sign((z[k,i]-z[k,j])))
                # e_up[k] : Boolean indicating if estimation at time k recover the right ordering between i and j
                                
            error_mse_lam[b] = np.mean(e_mse)
            error_up_lam[b] = np.mean(e_up)
                                
        # Mean error over bootsrap experiments
        error_mse[l] = np.mean(error_mse_lam)
        error_up[l] = np.mean(error_up_lam)
    
    # Select lambda that minimizes the MSE error
    index_mse = max(idx for idx, val in enumerate(error_mse) if val == np.min(error_mse[~np.isnan(error_mse)]))
    lam_mse = lambda_list[index_mse]
    
    # Select lambda that minimizes the Upsets error
    index_up = max(idx for idx, val in enumerate(error_up) if val == np.min(error_up[~np.isnan(error_up)]))
    lam_up = lambda_list[index_up]
                                
    # Compute corresponding estimate z
    Y_vec = ls.obs_transync(Y,A)
    Q = graph.diag_incidence(A)
    Lv = Q@Q.T
    z_mse = scipy.sparse.linalg.lsqr(Lv+lam_mse*E.T@E,Q@Y_vec)[0]
    z_mse = z_mse.reshape((N_out,N),order='F')
    
    z_up = scipy.sparse.linalg.lsqr(Lv+lam_up*E.T@E,Q@Y_vec)[0]
    z_up = z_up.reshape((N_out,N),order='F')
    
    # Center the estimate
    for k in range(N_out):
        z_mse[k,:] = z_mse[k,:]-np.mean(z_mse[k,:])
        z_up[k,:] = z_up[k,:]-np.mean(z_up[k,:])
    
    return lam_mse,z_mse,lam_up,z_up,error_up,error_mse

def cv_dls_transync_up(Y,A,E,lambda_list,num_loocv = 200):
    
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
            Y_vector = ls.obs_transync(Y_loocv,A_loocv)
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
    Y_vec = ls.obs_transync(Y,A)
    Q = graph.diag_incidence(A)
    Lv = Q@Q.T
    
    z_up = scipy.sparse.linalg.lsqr(Lv+lam_up*E.T@E,Q@Y_vec)[0]
    z_up = z_up.reshape((N_out,N),order='F')
    
    # Center the estimate
    for k in range(N_out):
        z_up[k,:] = z_up[k,:]-np.mean(z_up[k,:])
    
    return lam_up,z_up,error_up


def cv_dls_transync_mse(Y,A,E,lambda_list,num_loocv = 200):
    
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
            Y_vector = ls.obs_transync(Y_loocv,A_loocv)
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
    Y_vec = ls.obs_transync(Y,A)
    Q = graph.diag_incidence(A)
    Lv = Q@Q.T
    z_mse = scipy.sparse.linalg.lsqr(Lv+lam_mse*E.T@E,Q@Y_vec)[0]
    z_mse = z_mse.reshape((N_out,N),order='F')
    
    # Center the estimate
    for k in range(N_out):
        z_mse[k,:] = z_mse[k,:]-np.mean(z_mse[k,:])
    
    return lam_mse,z_mse,error_mse




def loocv_ls_btl(Y,A,E,lambda_list,num_loocv = 200):
    
    N_out,N = np.shape(A)[:2]
    indices = []
    N_comp = np.zeros(N_out)

    for k in range(N_out):
        indices.append(np.transpose(np.nonzero(A[k,:,:]))) 
        # Each element of the list is the array of all (i,j) possible combinations to choose from at time k
        N_comp[k] = np.shape(indices[k])[0] # total number of comparisons at time k
    
    error = np.zeros(len(lambda_list))
    for l,lam in enumerate(lambda_list):
        error_lam = np.zeros(num_loocv)
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
            Y_vector = ls.obs_transync(Y_loocv,A_loocv)
            Q = graph.diag_incidence(A_loocv)
            Lv = Q@Q.T 
            z = (scipy.sparse.linalg.lsqr(Lv+lam*E.T@E,Q@Y_vector)[0]).reshape((N_out,N),order='F')
            
            # Center and compare the estimate of probabilities and the observed probabilities for the items we remove
            e = np.zeros(N_out)
            for k in range(N_out):
                # Center
                z[k,:] = z[k,:] - np.mean(z[k,:])
                # Get weights w = exp(z)
                w = np.exp(z[k,:])
                # Compute estimation error on test values
                i,j = test_set[k,:]
                prob = w[i]/(w[i]+w[j])
                e[k] = (prob-Y[k,i,j])**2
                                
            error_lam[b] = np.mean(e)
                                
        # Mean error over bootsrap experiments
        error[l] = np.mean(error_lam)
    
    # Select lambda that minimizes the error
    index = max(idx for idx, val in enumerate(error) if val == np.min(error[~np.isnan(error)]))
    lam_star = lambda_list[index]
                                
    # Compute corresponding estimate z
    Y_vec = ls.obs_transync(Y,A)
    Q = graph.diag_incidence(A)
    Lv = Q@Q.T
    z = (scipy.sparse.linalg.lsqr(Lv+lam_star*E.T@E,Q@Y_vec)[0]).reshape((N_out,N),order='F')
    
    # Center the estimate
    for k in range(N_out):
        z[k,:] = z[k,:]-np.mean(z[k,:])

    return lam_star,z


### Projection method ###
def cv_dproj_transync(Y,A,E,V_E,eigs_E,tau_list,num_loocv = 200):
    
    N_out,N = np.shape(A)[:2]
    T = N_out-1
    indices = []
    N_comp = np.zeros(N_out)

    for k in range(N_out):
        indices.append(np.transpose(np.nonzero(A[k,:,:]))) 
        # Each element of the list is the array of all (i,j) possible combinations to choose from at time k
        N_comp[k] = np.shape(indices[k])[0] # total number of comparisons at time k
    
    error_mse = np.zeros(len(tau_list))
    error_up = np.zeros(len(tau_list))
    for l,tau in enumerate(tau_list):
        error_mse_tau = np.zeros(num_loocv)
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
            Y_vector = ls.obs_transync(Y_loocv,A_loocv)
            Q = graph.diag_incidence(A_loocv)
            Lv = Q@Q.T 
            
            z = scipy.sparse.linalg.lsqr(Lv,Q@Y_vector)[0]
            z_proj = (V_tau@V_tau.T@z).reshape((N_out,N),order='F')
            z = z.reshape((N_out,N),order='F')
            # Compare the estimate of strengths difference and the observed strengths difference for the items we remove
            e_mse = np.zeros(N_out)
            e_up = np.zeros(N_out)
            for k in range(N_out):
                # Center
                z_proj[k,:] = z_proj[k,:] - np.mean(z_proj[k,:])
                # Compare to test value
                i,j = test_set[k,:]
                e_mse[k] = (Y[k,i,j]-(z[k,i]-z[k,j]))**2
                e_up[k] = (np.sign(Y[k,i,j]) != np.sign((z[k,i]-z[k,j]))) 
                # e_up[k] : Boolean indicating if estimation at time k recover the right ordering between i and j
                                
            error_mse_tau[b] = np.mean(e_mse)
            error_up_tau[b] = np.mean(e_up)
                                
        # Mean error over bootsrap experiments
        error_mse[l] = np.mean(error_mse_tau)
        error_up[l] = np.mean(error_up_tau)
    
    # Select tau that minimizes the MSE
    index_mse = max(idx for idx, val in enumerate(error_mse) if val == np.min(error_mse[~np.isnan(error_mse)]))
    tau_mse = tau_list[index_mse]
    
    # Select tau that minimizes the Upsets
    index_up = max(idx for idx, val in enumerate(error_up) if val == np.min(error_up[~np.isnan(error_up)]))
    tau_up = tau_list[index_up]
                                
    # Compute corresponding estimate z
    Y_vec = ls.obs_transync(Y,A)
    Q = graph.diag_incidence(A)
    Lv = Q@Q.T
    z = (scipy.sparse.linalg.lsqr(Lv,Q@Y_vec)[0])
    
    # Projection
    # Compute projection matrix for this value of tau
    V_mse = V_E[:,eigs_E <= tau_mse]
    V_up = V_E[:,eigs_E <= tau_up]
    # V_tau is used in the projection method
            
    z_mse = (V_mse@V_mse.T@z).reshape((N_out,N),order='F')
    z_up = (V_up@V_up.T@z).reshape((N_out,N),order='F')
    
    # Center
    for k in range(N_out):
        z_mse[k,:] = z_mse[k,:] - np.mean(z_mse[k,:])
        z_up[k,:] = z_up[k,:] - np.mean(z_up[k,:])
    
    return tau_mse,z_mse,tau_up,z_up,error_up,error_mse

def cv_dproj_transync_up(Y,A,E,V_E,eigs_E,tau_list,num_loocv = 200):
    
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
            Y_vector = ls.obs_transync(Y_loocv,A_loocv)
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
    Y_vec = ls.obs_transync(Y,A)
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
    
    return tau_up,z_up,error_up

def cv_dproj_transync_mse(Y,A,E,V_E,eigs_E,tau_list,num_loocv = 200):
    
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
            Y_vector = ls.obs_transync(Y_loocv,A_loocv)
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
    Y_vec = ls.obs_transync(Y,A)
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
    
    return tau_mse,z_mse,error_mse




def loocv_proj_btl(Y,A,E,V_E,eigs_E,tau_list,num_loocv = 200):
    
    N_out,N = np.shape(A)[:2]
    T = N_out-1
    indices = []
    N_comp = np.zeros(N_out)

    for k in range(N_out):
        indices.append(np.transpose(np.nonzero(A[k,:,:]))) 
        # Each element of the list is the array of all (i,j) possible combinations to choose from at time k
        N_comp[k] = np.shape(indices[k])[0] # total number of comparisons at time k
    
    error = np.zeros(len(tau_list))
    for l,tau in enumerate(tau_list):
        error_tau = np.zeros(num_loocv)
        
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
            Y_vector = ls.obs_transync(Y_loocv,A_loocv)
            Q = graph.diag_incidence(A_loocv)
            Lv = Q@Q.T 
            
            z = scipy.sparse.linalg.lsqr(Lv,Q@Y_vector)[0]
            z_proj = (V_tau@V_tau.T@z).reshape((N_out,N),order='F')
            z = z.reshape((N_out,N),order='F')
            # Compare the estimate of strengths difference and the observed strengths difference for the items we remove
            
            # Compare the estimate of probabilites and the observed probabilites for the items we remove
            e=np.zeros(N_out)
            for k in range(N_out):
                # Center
                z_proj[k,:] = z_proj[k,:] - np.mean(z_proj[k,:])
                # Get weights w = exp(z)
                w = np.exp(z_proj[k,:])
                # Compute estimation error on test values
                i,j = test_set[k,:]
                prob = w[i]/(w[i]+w[j])
                e[k] = (prob-Y[k,i,j])**2
                                
            error_tau[b] = np.mean(e)
                                
        # Mean error over bootsrap experiments
        error[l] = np.mean(error_tau)
    
    # Select tau that minimizes the error
    index = max(idx for idx, val in enumerate(error) if val == np.min(error[~np.isnan(error)]))
    tau_star = tau_list[index]
                                
    # Compute corresponding estimate z
    Y_vec = ls.obs_transync(Y,A)
    Q = graph.diag_incidence(A)
    Lv = Q@Q.T
    z = scipy.sparse.linalg.lsqr(Lv,Q@Y_vec)[0]
    
    # Projection
    # Compute projection matrix for this value of tau
    V_star = V_E[:,eigs_E <= tau_star]
    # V_tau is used in the projection method
    z_proj = (V_star@V_star.T@z).reshape((N_out,N),order='F')
    
    # Center
    for k in range(N_out):
        z_proj[k,:] = z_proj[k,:] - np.mean(z_proj[k,:])
    
    return tau_star,z_proj


### DRC Method ###
def loocv_drc(Y,A,delta_list,num_loocv = 20):
    '''
    t : time at which we want to recover the ranks
    '''
    
    N_out,N = np.shape(A)[:2]
    indices = []
    T = N_out-1
    N_comp = np.zeros(N_out)

    for k in range(N_out):
        indices.append(np.transpose(np.nonzero(A[k,:,:]))) 
        # Each element of the list is the array of all (i,j) possible combinations to choose from at time k
        N_comp[k] = np.shape(indices[k])[0] # total number of comparisons at time k
    
    error = np.zeros(len(delta_list))
    for l,delta in enumerate(delta_list):
        error_delta = np.zeros(num_loocv)
        
        for b in range(num_loocv):
            Y_loocv = Y.copy()
            A_loocv = A.copy()
            
            test_set = np.zeros((N_out,2),dtype=int)
            e = np.zeros(N_out) # e[k] = error for estimation at time k
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
            
            for k in range(N_out):
                # Fit model and compute prediction error
                z= np.log(drc.RC_dyn(k/T,Y_loocv,A_loocv, delta, tol=1e-12)) # vector of length N
                # Center
                z = z-np.mean(z)
                # Error
                w = np.exp(z)
                prob = w[i]/(w[i]+w[j])
                e[k] = (prob-Y[k,i,j])**2
            
                               
            error_delta[b] = np.mean([x for x in e if np.isnan(x)==False])
                                
        # Mean error over bootsrap experiments for this value of delta
        error[l] = np.mean([x for x in error_delta if np.isnan(x)==False])
    
    # Select lambda that minimizes the error
    index = max(idx for idx, val in enumerate(error) if val == np.min(error[~np.isnan(error)]))
    delta_star = delta_list[index]
    z = np.zeros((N_out,N))
    for k in range(N_out):
        z[k,:] = np.log(drc.RC_dyn(k/T,Y,A, delta_star, tol=1e-12)) 
        # Center
        z[k,:] = z[k,:] - np.mean(z[k,:])
    
    return delta_star,z


### MLE Method ###

def loocv_mle_btl(Y,A,h_list,num_loocv = 20):
    
    N_out,N = np.shape(A)[:2]
    indices = []
    N_comp = np.zeros(N_out)
    
    data_MLE = np.transpose(Y,(0,2,1))
    T_list = np.arange(0,1,1/N_out)

    for k in range(N_out):
        indices.append(np.transpose(np.nonzero(A[k,:,:]))) 
        # Each element of the list is the array of all (i,j) possible combinations to choose from at time k
        N_comp[k] = np.shape(indices[k])[0] # total number of comparisons at time k
    
    error = np.zeros(len(h_list))
    for l,h in enumerate(h_list):
        error_h = np.zeros(num_loocv)
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
            data = np.transpose(Y_loocv,(0,2,1))
            ks_data = mle.kernel_smooth(data,h,T_list)
            z = mle.gd_bt(data = ks_data,verbose=True)[1]
            #z = z.reshape((N_out,N),order='F')
            # Center and compare the estimate of probabilities and the observed probabilities for the items we remove
            e = np.zeros(N_out)
            for k in range(N_out):
                # Center
                z[k,:] = z[k,:] - np.mean(z[k,:])
                # Get weights w = exp(z)
                w = np.exp(z[k,:])
                # Compute estimation error on test values
                i,j = test_set[k,:]
                prob = w[i]/(w[i]+w[j])
                e[k] = (prob-Y[k,i,j])**2
                                
            error_h[b] = np.mean(e)
                                
        # Mean error over bootsrap experiments
        error[l] = np.mean(error_h)
    # Select lambda that minimizes the error
    index = max(idx for idx, val in enumerate(error) if val == np.min(error[~np.isnan(error)]))
    h_star = h_list[index]
                                
    # Compute corresponding estimate z
    ks_data = mle.kernel_smooth(data_MLE,h_star,T_list)
    z = mle.gd_bt(data = ks_data,verbose=True)[1]
    #z = z.reshape((N_out,N),order='F')
    
    # Center the estimate
    for k in range(N_out):
        z[k,:] = z[k,:]-np.mean(z[k,:])

    return h_star,z

