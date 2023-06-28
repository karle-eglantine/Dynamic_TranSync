import sys, os, csv, importlib
import numpy as np
import scipy
import scipy.linalg as spl
import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import simulation_module as sim
#import grad_module as model
import mle_module as mle
import loocv_module as cv
import ls_module as ls
import graph_module as graph


## Transync setup ##

def get_single_round_matrix_transync(rnd_num, nfl_data_dir, season):
    """
    Gets the pairwise numpy array of win/loss across teams for a single
       round in a season. pwise_diff[i,j] = score[i] - score[j] at this round.
    """
    fname = "round" + "_" + str(rnd_num).zfill(2) + ".csv"
    fpath = os.path.join(nfl_data_dir, str(season), fname)
    rnd_df = pd.read_csv(fpath)
    
    # Matrix of data Y
    Y = rnd_df.pivot(index='team', columns='team_other',values='diff').values
    Y[np.isnan(Y)] = 0
    #Y = Y/np.max(Y)
    
    # Adjacency matrix A
    A = rnd_df.pivot(index='team', columns='team_other',values='diff').values
    A[~np.isnan(A)] = 1
    A[np.isnan(A)] = 0
    
    return Y,A


def get_ranks_transync(data_dir, all_season, team_id, all_rnds,elo_all,num_loocv = 20,size_loocv = 40,dls_flag=True,dproj_flag = True, elo_flag = False):
    
    N = len(team_id)
    T = len(all_rnds)*len(all_season)
    nb_rnds = len(all_rnds)
    Y = np.zeros((T,N,N))
    A = np.zeros((T,N,N))
    
    lambda_list = np.linspace(0,T**(4/3),size_loocv) # We try for lam = T^l, with l in lamda_list
    tau_list = np.linspace(0,T**(-4/3),size_loocv)
    
    for i,season in enumerate(all_season):
        for j,rnd in enumerate(all_rnds):
            Y[i*nb_rnds+j,:,:],A[i*nb_rnds+j,:,:] = get_single_round_matrix_transync(rnd_num=rnd, nfl_data_dir=data_dir, season=season)
    # T-N-N arrays, T = total nb of rounds accross all seasons.
    # Y = score differences for each match
    result = []
    
    # Compute penalty matrix E
    E = ls.penalty_E(N,T)
    
    ### ELO Ratings ###
    if elo_flag:
        df_elo = team_id[['name']].copy()
        for i,season in enumerate(all_season):
            ranks_elo = get_elo_rank_season(elo_all,season)
            df_elo.insert(1+i,'Rank at time '+str(i),ranks_elo)
            df_elo.rename(columns = {'name':'ELO'}, inplace = True)
        result.append(df_elo)
    
    ### LS Method ###
    if dls_flag:
        # Tune parameter lambda
        lambda_dls,z_dls = cv.loocv_ls_transync(Y,A,E,lambda_list,num_loocv)
        # z_ls is vector of length rnds*teams = 16*32
        z_dls_tab = z_dls.reshape((T,N),order='F')
        
        df_dls = team_id[['name']].copy()
        for k in range(T):
            ranks_dls = ss.rankdata(-z_dls_tab[k,:],method='average') # Ranks at time t.
            df_dls.insert(1+k,'Rank at time '+str(k),ranks_dls)
            df_dls.rename(columns = {'name':'DLS'}, inplace = True)
            
        kt_dls = scipy.stats.kendalltau(ranks_dls,ranks_elo)
        l_dls = [z_dls,df_dls,lambda_dls,kt_dls]
        
        result.append(l_dls)
        
    ### Projection Method ###
    if dproj_flag:
        # get eigenvectors of E.T@E
        with open("eigenpairs_E/eigenvectors_E_N"+str(N)+"_T"+str(T)+".txt", "rb") as v:
            V_E = pickle.load(v)
        with open("eigenpairs_E/eigenvalues_E_N"+str(N)+"_T"+str(T)+".txt", "rb") as e:
            eigs_E = pickle.load(e)
        
        # Tune parameter tau
        tau_dproj,z_dproj,z_ls = cv.loocv_proj_transync(Y,A,E,V_E,eigs_E,tau_list,num_loocv)
        # z_proj is vector of length rnds*teams = 16*32
        z_dproj_tab = z_dproj.reshape((T,N),order='F')
        z_ls_tab = z_ls.reshape((T,N),order='F')
        
        df_dproj = team_id[['name']].copy()
        for k in range(T):
            ranks_dproj = ss.rankdata(-z_dproj_tab[k,:],method='average') # Ranks for the chosen season.
            df_dproj.insert(1+k,'Rank at time '+str(k),ranks_dproj)
            df_dproj.rename(columns = {'name':'DProj'}, inplace = True)
        kt_dproj = scipy.stats.kendalltau(ranks_dproj,ranks_elo)
        l_dproj = [z_dproj,df_dproj,tau_dproj,kt_dproj]
        
        result.append(l_dproj)
        
        df_ls = team_id[['name']].copy()
        for k in range(T):
            ranks_ls = ss.rankdata(-z_ls_tab[k,:],method='average') # Ranks for the chosen season.
            df_ls.insert(1+k,'Rank at time '+str(k),ranks_ls)
            df_ls.rename(columns = {'name':'LS'}, inplace = True)
        kt_ls = scipy.stats.kendalltau(ranks_ls,ranks_elo)
        l_ls = [z_ls,df_ls,kt_ls]
        
        result.append(l_ls)
        
        
    return result,Y,A

def get_ranks_transync_lambda(lam,tau,data_dir, all_season, team_id, all_rnds,elo_all,dls_flag=True,dproj_flag = True, elo_flag = False):
    
    N = len(team_id)
    T = len(all_rnds)*len(all_season)
    nb_rnds = len(all_rnds)
    Y = np.zeros((T,N,N))
    A = np.zeros((T,N,N))
    
    for i,season in enumerate(all_season):
        for j,rnd in enumerate(all_rnds):
            Y[i*nb_rnds+j,:,:],A[i*nb_rnds+j,:,:] = get_single_round_matrix_transync(rnd_num=rnd, nfl_data_dir=data_dir, season=season)
    # T-N-N arrays, T = total nb of rounds accross all seasons.
    # Y = score differences for each match
    result = []
    
    # Compute penalty matrix E
    E = ls.penalty_E(N,T)
    Y_vec = ls.obs_transync(Y,A)
    Q = graph.diag_incidence(A)
    Lv = Q@Q.T
    
    ### ELO Ratings ###
    if elo_flag:
        df_elo = team_id[['name']].copy()
        for i,season in enumerate(all_season):
            ranks_elo = get_elo_rank_season(elo_all,season)
            df_elo.insert(1+i,'Rank at time '+str(i),ranks_elo)
            df_elo.rename(columns = {'name':'ELO'}, inplace = True)
        result.append(df_elo)
    
    ### LS Method ###
    if dls_flag:
        z_dls = scipy.sparse.linalg.lsqr(Lv+lam*E.T@E,Q@Y_vec)[0]
        z_dls = z_dls.reshape((T,N),order='F')
        # Center the estimate
        for k in range(T):
            z_dls[k,:] = z_dls[k,:]-np.mean(z_dls[k,:])
        
        
        df_dls = team_id[['name']].copy()
        for k in range(T):
            ranks_dls = ss.rankdata(-z_dls[k,:],method='average') # Ranks at time t.
            df_dls.insert(1+k,'Rank at time '+str(k),ranks_dls)
            df_dls.rename(columns = {'name':'DLS'}, inplace = True)
        kt_dls = scipy.stats.kendalltau(ranks_dls,ranks_elo)[0]
        l_dls = [z_dls,df_dls,lam,kt_dls]
        
        result.append(l_dls)
        
    ### Projection Method ###
    if dproj_flag:
        # get eigenvectors of E.T@E
        with open("eigenpairs_E/eigenvectors_E_N"+str(N)+"_T"+str(T)+".txt", "rb") as v:
            V_E = pickle.load(v)
        with open("eigenpairs_E/eigenvalues_E_N"+str(N)+"_T"+str(T)+".txt", "rb") as e:
            eigs_E = pickle.load(e)
        
        # Tune parameter tau
        z_ls = (scipy.sparse.linalg.lsqr(Lv,Q@Y_vec)[0])
    
        # Projection
        # Compute projection matrix for this value of tau
        V = V_E[:,eigs_E <= tau]
        # V_tau is used in the projection method

        z_dproj = (V@V.T@z_ls).reshape((T,N),order='F')
        z_ls = z_ls.reshape((T,N),order='F')
        # Center
        for k in range(T):
            z_dproj[k,:] = z_dproj[k,:] - np.mean(z_dproj[k,:])
            z_ls[k,:] = z_ls[k,:] - np.mean(z_ls[k,:])
            # z_proj is vector of length rnds*teams = 16*32
        z_dproj_tab = z_dproj.reshape((T,N),order='F')
        z_ls_tab = z_ls.reshape((T,N),order='F')

        df_dproj = team_id[['name']].copy()
        for k in range(T):
            ranks_dproj = ss.rankdata(-z_dproj_tab[k,:],method='average') # Ranks for the chosen season.
            df_dproj.insert(1+k,'Rank at time '+str(k),ranks_dproj)
            df_dproj.rename(columns = {'name':'DProj'}, inplace = True)
        kt_dproj = scipy.stats.kendalltau(ranks_dproj,ranks_elo)[0]
        l_dproj = [z_dproj,df_dproj,tau,kt_dproj]
        
        result.append(l_dproj)
        
        df_ls = team_id[['name']].copy()
        for k in range(T):
            ranks_ls = ss.rankdata(-z_ls_tab[k,:],method='average') # Ranks for the chosen season.
            df_ls.insert(1+k,'Rank at time '+str(k),ranks_ls)
            df_ls.rename(columns = {'name':'LS'}, inplace = True)
        kt_ls = scipy.stats.kendalltau(ranks_ls,ranks_elo)[0]
        l_ls = [z_ls,df_ls,kt_ls]
        
        result.append(l_ls)
        
        
    return result,Y,A

### Number of upsets ###
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
        upsets.append(np.sum(np.sign(Y_vec) == np.sign(z_vec)))
    
    return upsets

def get_nb_upsets_btl(Y,A,z):
    """ 
    Y : T-N-N matrix of observations under TranSync model
    A : T-N-N adjacency matrix of observations
    z : N_out-N matrix where z[k,:] is the estimated strengths at time k.
    --------
    Return the vector of lenght nb_rnds that counts at each time the number of pairs
    """
    T= np.shape(A)[0]
    upsets = []
    w = np.exp(z)
    for t in range(T):
        Y_vec = []
        z_vec = []
        edges = graph.list_edges(A[t,:,:])
        for i in range(len(edges)):
            a,b = edges[i]
            Y_vec.append(Y[t,a,b]-1/2)
            z_vec.append(w[t,a]/(w[t,a]+w[t,b])-1/2)
        
        upsets.append(np.sum(np.sign(Y_vec) == np.sign(z_vec)))
    
    return upsets
    
    


### BTL Model ###

def get_single_round_matrix_btl(rnd_num, nfl_data_dir, season):
    """
    Gets the pairwise numpy array of win/loss across teams for a single
       round in a season. pwise_diff[i,j] = 1 if j won against i at this round.
    """
    fname = "round" + "_" + str(rnd_num).zfill(2) + ".csv"
    fpath = os.path.join(nfl_data_dir, str(season), fname)
    rnd_df = pd.read_csv(fpath)
    
    # Matrix of data Y
    Y = rnd_df.pivot(index='team', columns='team_other',values='diff').values
    Y[Y >= 0] = 0
    Y[Y < 0] = 1
    Y[np.isnan(Y)] = 0
    
    # Adjacency matrix A
    A = rnd_df.pivot(index='team', columns='team_other',values='diff').values
    A[~np.isnan(A)] = 1
    A[np.isnan(A)] = 0
    
    return Y,A


def get_ranks_btl(data_dir, all_season, team_id, all_rnds,elo_all,num_loocv = 20,ls_flag=True,proj_flag = True, elo_flag = False,drc_flag = True,mle_flag = True):
    
    N = len(team_id)
    T = len(all_rnds)*len(all_season)
    nb_rnds = len(all_rnds)
    Y = np.zeros((T,N,N))
    A = np.zeros((T,N,N))
    for i,season in enumerate(all_season):
        for j,rnd in enumerate(all_rnds):
            Y[i*nb_rnds+j,:,:],A[i*nb_rnds+j,:,:] = get_single_round_matrix_btl(rnd_num=rnd, nfl_data_dir=data_dir, season=season)
    # T-N-N arrays, T = total nb of rounds accross all seasons.
    # Y = score differences for each match
    result = []
    
    # Compute penalty matrix E
    E = ls.penalty_E(N,T)
    
    lambda_list = T**np.linspace(0,4/3,5) # We try for lam = T^l, with l in lamda_list
    tau_list = T**np.linspace(-4/3,0,5)
    
    ### ELO Ratings ###
    if elo_flag:
        df_elo = team_id[['name']].copy()
        for i,season in enumerate(all_season):
            ranks_elo = get_elo_rank_season(elo_all,season)
            df_elo.insert(1+i,'Rank at time '+str(i),ranks_elo)
            df_elo.rename(columns = {'name':'ELO'}, inplace = True)
        result.append(df_elo)
    
    ### LS Method ###
    if ls_flag:
        # Tune parameter lambda
        lambda_ls,z_ls = cv.loocv_ls_btl(Y,A,E,lambda_list,num_loocv = 20)
        # z_ls is vector of length rnds*teams = 16*32
        z_ls_tab = z_ls.reshape((T,N),order='F')
        
        df_ls = team_id[['name']].copy()
        for k in range(T):
            ranks_ls = ss.rankdata(-z_ls_tab[k,:],method='average') # Ranks at time t.
            df_ls.insert(1+k,'Rank at time '+str(k),ranks_ls)
            df_ls.rename(columns = {'name':'LS'}, inplace = True)
        l_ls = [z_ls,df_ls,lambda_ls]
        
        result.append(l_ls)
        
    ### Projection Method ###
    if proj_flag:
        # get eigenvectors of E.T@E
        with open("eigenpairs_E/eigenvectors_E_N"+str(N)+"_T"+str(T)+".txt", "rb") as v:
            V_E = pickle.load(v)
        with open("eigenpairs_E/eigenvalues_E_N"+str(N)+"_T"+str(T)+".txt", "rb") as e:
            eigs_E = pickle.load(e)
        
        # Tune parameter tau
        tau_proj,z_proj = cv.loocv_proj_btl(Y,A,E,V_E,eigs_E,tau_list,num_loocv = 20)
        # z_proj is vector of length rnds*teams = 16*32
        z_proj_tab = z_proj.reshape((T,N),order='F')
        
        df_proj = team_id[['name']].copy()
        for k in range(T):
            ranks_proj = ss.rankdata(-z_proj_tab[k,:],method='average') # Ranks for the chosen season.
            df_proj.insert(1+k,'Rank at time '+str(k),ranks_proj)
            df_proj.rename(columns = {'name':'Proj'}, inplace = True)
        l_proj = [z_proj,df_proj,tau_proj]
        
        result.append(l_proj)
        
    ### DRC Method ###
    if drc_flag:
        delta_list = np.linspace(1/2, T,5)
        df_rc = team_id[['name']].copy()
        delta_rc,z_rc = cv.loocv_drc(Y,A,delta_list,num_loocv)
        for k in range(T):
            ranks_rc = ss.rankdata(-z_rc[k,:],method='average')
            df_rc.insert(1+k,'Rank at time '+str(k),ranks_rc)
            df_rc.rename(columns = {'name':'RC'}, inplace = True)
        l_rc = [z_rc,df_rc,delta_rc]
        
        result.append(l_rc)
        
    #### MLE MEthod ###
    if mle_flag:
        h_list = np.linspace(0.5, 0.05, 5)
        h_mle,z_mle = cv.loocv_mle_btl(Y,A,h_list,num_loocv = 20)
        # z_ls is vector of length rnds*teams = 16*32
        z_mle_tab = z_mle.reshape((T,N),order='F')
        
        df_mle = team_id[['name']].copy()
        for k in range(T):
            ranks_mle = ss.rankdata(-z_mle_tab[k,:],method='average') # Ranks at time t.
            df_mle.insert(1+k,'Rank at time '+str(k),ranks_mle)
            df_mle.rename(columns = {'name':'MLE'}, inplace = True)
        l_mle = [z_mle,df_mle,h_mle]
        
        result.append(l_mle)
        
        
    return result,Y,A
    

### ELO Method ###
        
def get_elo_rank_season(elo_all, season):
    elo_season = elo_all.iloc[np.where(elo_all['season'] == season)]
    elo_season = elo_season[pd.isnull(elo_season['playoff'])]
    a = elo_season[['team1','elo1_post']]
    a.columns = ['team','elo']
    a = a.reset_index()
    b = elo_season[['team2','elo2_post']]
    b.columns = ['team','elo']
    b = b.reset_index()

    c = pd.concat([a,b])
    c = c.sort_values(by = ['index'])    
    d = c.groupby(by = ['team']).last()
    
    x = d.index.values
    x[np.where(x == 'LAR')] = 'STL'
    x[np.where(x == 'LAC')] = 'SD'
    x[np.where(x == 'JAX')] = 'JAC'
    x[np.where(x == 'WSH')] = 'WAS'
    
    elo_rank = ss.rankdata(-d['elo'])
    
    return elo_rank





