# Dynamic_TranSync

This repository contains the code for the paper Dynamic Ranking and Translation Synchronization, Araya et al., 2023. https://arxiv.org/abs/2207.01455

Files

 + Plots_synthetic_data.ipynb is a notebook that contains the scripts to generate Figures 1 to 5 and C1, C2 (experiments on synthetic data). By default, the notebook runs toy examples to provide a good understanding of the functions to the user. The parameters used in the paper need to be uncommented to reproduce the figures of the article.
 + Netflix_analysis.ipynb is a notebook that contains the analysis of the Netflix Prize dataset, leading to Table 1 and Figure 6a.
 + EPL_analysis.ipynb is a notebook that contains the analysis of the English Premier League dataset, leading to Table 2 and Figure 6b.
 + dyn_transync.yml: conda environment
 + python is a directory containing all the tools used in the notebooks. It contains in particular:
   - transync.py contains a function to perform analysis under the Dynamic TranSync model
   - btl_analysis.py contains a function to perform analysis under the Dynamic BTL model
   - performance_dls_vs_lambda.py contains a function to perform DLS method for different parameters $\lambda$.
   - performance_dproj_vs_tau.py contains a function to perform Dproj method for different parameters $\tau$.
   - eigs_L_lam.py contains a function to compute eigenvalues of $L(\lambda)$.
   - modules is a directory containing different technical tools.
