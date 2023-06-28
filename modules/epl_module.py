import sys, os, csv, importlib
import numpy as np
import scipy
import scipy.linalg as spl
import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import ls_module as ls
import graph_module as graph


## Transync setup ##


def get_ranks_transync(lam,tau,Y,A,teams,ls_flag = True,dls_flag=True,dproj_flag = True):
    
    T,N = np.shape(Y)[:2]
    # T-N-N arrays, T = total nb of rounds accross all seasons.
    # Y = score differences for each match
    result = []
    
    # Compute penalty matrix E
    E = ls.penalty_E(N,T)
    Y_vec = ls.obs_transync(Y,A)
    Q = graph.diag_incidence(A)
    Lv = Q@Q.T
    
    ### Naive LS Estimator ###
    if ls_flag:
        z_ls = (scipy.sparse.linalg.lsqr(Lv,Q@Y_vec)[0])
        z_ls = z_ls.reshape((T,N),order='F')
        # Center
        for k in range(T):
            z_ls[k,:] = z_ls[k,:] - np.mean(z_ls[k,:])
        
        df_ls = pd.DataFrame({'Teams': teams})
        for k in range(T):
            ranks_ls = ss.rankdata(-z_ls[k,:],method='average') # Ranks for the chosen season.
            df_ls.insert(1+k,'Rank at time '+str(k),ranks_ls)
            df_ls.rename(columns = {'name':'LS'}, inplace = True)
            df_ls.set_index('Teams')
        l_ls = [z_ls,df_ls]
        
        result.append(l_ls)
    
    ### DLS Method ###
    if dls_flag:
        z_dls = scipy.sparse.linalg.lsqr(Lv+lam*E.T@E,Q@Y_vec)[0]
        z_dls = z_dls.reshape((T,N),order='F')
        # Center the estimate
        for k in range(T):
            z_dls[k,:] = z_dls[k,:]-np.mean(z_dls[k,:])
        
        df_dls = pd.DataFrame({'Teams': teams})
        for k in range(T):
            ranks_dls = ss.rankdata(-z_dls[k,:],method='average') # Ranks at time t.
            df_dls.insert(1+k,'Rank at time '+str(k),ranks_dls)
            df_dls.rename(columns = {'name':'DLS'}, inplace = True)
            df_dls.set_index('Teams')
        l_dls = [z_dls,df_dls,lam]
        
        result.append(l_dls)
        
    ### Projection Method ###
    if dproj_flag:
        # get eigenvectors of E.T@E
        with open("eigenpairs_E/eigenvectors_E_N"+str(N)+"_T"+str(T)+".txt", "rb") as v:
            V_E = pickle.load(v)
        with open("eigenpairs_E/eigenvalues_E_N"+str(N)+"_T"+str(T)+".txt", "rb") as e:
            eigs_E = pickle.load(e)
        
        # Tune parameter tau
        z_ls = scipy.sparse.linalg.lsqr(Lv,Q@Y_vec)[0]
    
        # Projection
        # Compute projection matrix for this value of tau
        V = V_E[:,eigs_E < tau]
        # V_tau is used in the projection method

        z_dproj = (V@V.T@z_ls).reshape((T,N),order='F')
        # Center
        for k in range(T):
            z_dproj[k,:] = z_dproj[k,:] - np.mean(z_dproj[k,:])
            # z_proj is vector of length rnds*teams = 16*32

        df_dproj = pd.DataFrame({'Teams': teams})
        for k in range(T):
            ranks_dproj = ss.rankdata(-z_dproj[k,:],method='average') # Ranks for the chosen season.
            df_dproj.insert(1+k,'Rank at time '+str(k),ranks_dproj)
            df_dproj.rename(columns = {'name':'DProj'}, inplace = True)
            df_dproj.set_index('Teams')
        l_dproj = [z_dproj,df_dproj,tau]
        
        result.append(l_dproj)
        
        
    return result

def get_nb_upsets_transync(Y,A,z):
    """ 
    z : N_out-N matrix where z[k,:] is the estimated strengths at time k.
    --------
    Return the vector of lenght nb_rnds that counts at each time the number of pairs for which the winner is wrongly estimated
    """
    T= np.shape(A)[0]
    upsets = []
    for t in range(T):
        Y_vec = []
        z_vec = []
        edges = graph.list_edges(A[t,:,:])
        for i in range(len(edges)):
            a,b = edges[i]
            Y_vec.append(Y[t,a,b])
            z_vec.append(z[t,a]-z[t,b])
        # Is consider an upset every pair for which the estimated winner is not the observed winner
        upsets.append(np.sum(np.sign(Y_vec) != np.sign(z_vec)))
    
    return upsets

def get_mean_upsets_transync(Y,A,z):
    """ 
    z : N_out-N matrix where z[k,:] is the estimated strengths at time k.
    --------
    Return the vector of lenght nb_rnds that counts at each time the number of pairs for which the winner is wrongly estimated
    """
    T= np.shape(A)[0]
    upsets = []
    for t in range(T):
        Y_vec = []
        z_vec = []
        edges = graph.list_edges(A[t,:,:])
        for i in range(len(edges)):
            a,b = edges[i]
            Y_vec.append(Y[t,a,b])
            z_vec.append(z[t,a]-z[t,b])
        # Is consider an upset every pair for which the estimated winner is not the observed winner
        upsets.append(np.mean(np.sign(Y_vec) != np.sign(z_vec)))
    
    return upsets

def get_mse_upsets_transync(Y,A,z):
    """ 
    z : N_out-N matrix where z[k,:] is the estimated strengths at time k.
    --------
    Return the vector of lenght nb_rnds that counts at each time the number of pairs for which the winner is wrongly estimated
    """
    T= np.shape(A)[0]
    mse_upsets = []
    for t in range(T):
        edges = graph.list_edges(A[t,:,:])
        x = np.zeros(len(edges))
        for i in range(len(edges)):
            a,b = edges[i]
            x = Y[t,a,b]-(z[t,a]-z[t,b])
        # Is consider an upset every pair for which the estimated winner is not the observed winner
        mse_upsets.append(np.linalg.norm(x)**2)
    
    return mse_upsets