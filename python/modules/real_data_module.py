# This script is written by E. Karlé to conduct the experiments of the article Dynamic Ranking and Translation Synchornization https://arxiv.org/pdf/2207.01455.pdf
#
# This module contains functions used to analyse real datasets

import sys, os, csv, importlib
import numpy as np
import scipy
import scipy.linalg as spl
import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import tools_module as tools
import graph_module as graph

# Computation of MSE and Number of Upsets on real data
def get_mean_nb_upsets(Y,A,z):
    """ 
    Y: T-N-N observation matrix
    A: T-N-N adjacency matrix
    z: T-N matrix where z[k,:] is the estimated strengths at time k.
    --------
    Return the mean number (across all the timepoints) of misranked pairs in our estimation with respect to the observation
    """
    T= np.shape(A)[0]
    upsets = [] # List of number of misranked pairs at each time
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
    
    return np.mean(upsets)

def get_mse(Y,A,z):
    """ 
    Y: T-N-N observation matrix
    A: T-N-N adjacency matrix
    z: T-N matrix where z[k,:] is the estimated strengths at time k.
    --------
    Return the mean MSE (across all timepoints) between the observed score differences (Y) and our estimated score differences
    """
    T= np.shape(A)[0]
    mse = []
    for t in range(T):
        edges = graph.list_edges(A[t,:,:])
        x = np.zeros(len(edges))
        for i in range(len(edges)):
            a,b = edges[i]
            x[i] = Y[t,a,b]-(z[t,a]-z[t,b])
        mse.append(np.linalg.norm(x)**2)
    
    return np.mean(mse)

### Tools for the EPL dataset
# Construction of observation and adjacency dataframes corresponding to one week in a season
def get_single_round_matrix(data,teams):
    """
    Construct dataframes of the teams that played at a given week in a given season as well as the observed goal differences
    """
    
    # Dataframe of goal differences
    Y = pd.DataFrame(index=teams,columns=teams)
    Y.fillna(0,inplace=True)
    # Dataframe of played games
    A = pd.DataFrame(index=teams,columns=teams)
    A.fillna(0,inplace=True)
    
    for i in data.HomeTeam:
        for j in data.AwayTeam:
            s = data.Score.loc[(data.HomeTeam==i) & (data.AwayTeam ==j)].values # s = goals difference in game between teams i and j
            if s.size >0:
                Y.loc[[i],[j]] += s
                Y.loc[[j],[i]] -= s
                
                A.loc[[i],[j]] = 1
                A.loc[[i],[j]] = 1
    
    return Y,A


