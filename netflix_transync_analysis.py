import sys, os, csv, importlib
import numpy as np
import scipy as sc
import scipy.linalg as spl
import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt
import random
import pickle

import sys
sys.path.append('modules')

import netflix_module as netflix

# Load data
movie_id = pd.read_csv("netflix_data/movie_titles.csv")
with open("netflix_data/y_transync.txt", "wb") as y:
    Y = pickle.load(y)
with open("netflix_data/a_transync.txt", "wb") as a:
    A = pickle.load(a)


random.seed(0)
np.random.seed(0)

result = netflix.get_ranks_transync(Y,A,num_loocv = 20,size_loocv = 40,ls_flag=True,proj_flag = True)

l_ls,l_proj= result
z_ls,df_ls,lambda_ls = l_ls
z_proj,df_proj,tau_proj = l_proj

# Save results with value of delta_star in title
df_ls.to_csv('netflix_ranks_ls_transync.csv')
df_proj.to_csv('netflix_ranks_proj_transync.csv')