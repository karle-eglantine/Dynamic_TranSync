# Dynamic_TranSync

This repository contains the code for the paper Dynamic Ranking and Translation Synchronization, Araya et al., 2023 (https://arxiv.org/abs/2207.01455)

Files

 + Plots_synthetic_data.ipynb is a notebook that contains the scripts to generate Figures 1 to 5 and C1, C2 (experiments on synthetic data).
 + Netflix_analysis.ipynb is a notebook that contains the analysis of the Netflix Prize dataset, leading to Table 1 and Figure 6a.
 + EPL_analysis.ipynb is a notebook that contains the analysis of the English Premier League dataset, leading to Table 2 and Figure 6b.
 + transync.py contains a function to perform analysis under the Dynamic TranSync model
 + btl_analysis.py contains a function to perform analysis under the Dynamic BTL model
 + performance_dls_vs_lambda.py contains a function to perform DLS method for different parameters $\lambda$.
 + performance_dproj_vs_tau.py contains a function to perform Dproj method for different parameters $\tau$.
 + eigs_L_lam.py contains a function to compute eigenvalues of $L(\lambda)$.
 + modules is a directory containing useful python scripts 
