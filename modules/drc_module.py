import numpy as np
import scipy

import graph_module as graph

def update_neighborhood(neigh):
    T = np.shape(neigh)[0]
    idx = np.nonzero(neigh)[0]
    if np.shape(idx)[0] == T: # The neighborhood is already equal to the complete grid
        return idx
    else:
        new_neigh = neigh
        if idx[0] == 0: # Can't complete the neighborhood on the left : already include time t = 0
            new_neigh[idx[-1]+1] = True # Add one point on the right
        else:
            new_neigh[idx[0]-1] = True # Add a point on the left
            if idx[-1]!= T-1: # Can also add a point on the right
                new_neigh[idx[-1]+1] = True
    return np.nonzero(new_neigh)[0]


def get_delta_drc(A,vec_delta,output_grid):
    '''
    Generate list of T+1 adjacency matrices and vector of values of delta such that for any t in the output_grid, G_delta(t) is connected
    ---------------------
    Input :
    A : T-N-N array such that A[t,:,:] is the adjacency of graph at time t. We ensured that union of all the graphs are connected
    vec_delta : vector of same length as output_grid, initialized to be the constant vector with value delta^* = c_delta*T**(2/3)/(N*L**1/3)
    output_grid : list of times at which we want to recover the ranks
    
    Output:
    vec_delta : updated values of delta, for wich G_delta(t) is connected (t in output_grid)
    '''
    T,N = np.shape(A)[:2]
    grid = np.arange(0,1+1/(2*T),1/T)
    for i,t in enumerate(output_grid):
        A_delta = graph.union_graph(A,vec_delta[i],t)
        N_delta = graph.neighborhood(t,vec_delta[i],T)
        while sum(N_delta) == 0 or not graph.connected(A_delta):
            # Add graphs on each side of the neighborhood
            idx = update_neighborhood(N_delta) # List of indexes of points of the grid in the neighborhood. We add (if possible) one grid point on the left and on the right to the old neighborhood.
            vec_delta[i] = T*max(t-grid[idx[0]],grid[idx[-1]]-t)
            A_delta = graph.union_graph(A,vec_delta[i],t)
    return vec_delta

def transition_matrix(A_delta,Y,N_delta):
    '''
    Compute the transition matrix of the union graph at time t
    Input : 
        A_delta : adajcency matrix of the union graph
        Y : (T+1)-N-N array containig the pairwise comparison information, averaged on the L workers at each time
        N_delta : neighborhood
    Output : N-N arary
    '''
    N = np.shape(Y)[1]
    transition_matrix = np.zeros((N,N))
    d = 2*N*graph.graph_proba(A_delta)
    for i in range(N):
        for j in range(N):
            if (i != j) & (A_delta[i,j] != 0):
                transition_matrix[i,j] = sum(Y[N_delta,i,j])/(d*A_delta[i,j])
        transition_matrix[i,i] = 1-np.sum(transition_matrix[i,:])
    return transition_matrix

def RC_dyn(t,Y,A,delta,tol = 1e-12):
    '''
    get the estimator pihat_RC(t)
    ------------
    Input :
    Y: (T+1)-N-N array containig the pairwise comparison information, averaged on the L workers at each time
    A: (T+1)-N-N array, A[t,:,:] is the adjacency matrix of data at time t
    tol = tolerance to approximate the eigenvalues equal to 1. 
    Output :
    If union graph disconnected
    '''
    T,N = np.shape(A)[:2]
    N_delta = graph.neighborhood(t,delta,T-1)
    A_delta = graph.union_graph(A,delta,t)
    P = transition_matrix(A_delta,Y,N_delta)
    eigval,eigvec = scipy.linalg.eig(P,left=True,right=False)
    pi_RC = eigvec[:,abs(eigval - 1) < tol][:,0]
    return pi_RC/sum(pi_RC)

### LOOCV ###

def loocv_rc(Y,A,delta_list,num_loocv = 200,t=1):
    '''
    t : time at which we want to recover the ranks
    '''    
    T,N = np.shape(A)[:2]
    indices = np.transpose(np.nonzero(A)) # Array of all (t,i,j) possible combinations to choose from
    N_comp = np.shape(indices)[0] # total number of comparisons
    
    error = np.zeros(len(delta_list))
    for l,delta in enumerate(delta_list):
        error_delta = np.zeros(num_loocv)
        for k in range(num_loocv):
            Y_loocv = Y.copy()
            A_loocv = A.copy()
        
            rand_match = np.random.randint(N_comp) # random number between 0 and Total number of comparisons
            rand_index = indices[rand_match,:] # Select the tuple (t,i,j) corresponding to the rand_match comparison
            s,i,j = tuple(rand_index)
        
            # Remove the test value from the data
            Y_loocv[s,i,j] = max(Y_loocv[s,i,j]-1,0) # if all observations at these time where 0, then Y[t,i,j] stays 0.
            Y_loocv[s,j,i] = max(Y_loocv[s,j,i]-1,0)
            A_loocv[s,i,j] = 0
            A_loocv[s,j,i] = 0  
            
            # Fit model and compute prediction error
            pi = sim.RC_dyn(s/T,Y_loocv,A_loocv, delta, tol=1e-12) # vector of length N
            prob = pi[j]/(pi[i]+pi[j])
            error_delta[k] = np.linalg.norm(prob-Y[s,i,j])
        error[l] = np.mean(error_delta)

    index = max(idx for idx, val in enumerate(error) if val == np.min(error[~np.isnan(error)]))
    delta_star = delta_list[index]
    pi_star = RC_dyn(t,Y,A, delta_star, tol=1e-12)
    
    return delta_star,pi_star

