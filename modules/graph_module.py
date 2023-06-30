# This script is written by E. Karlé to conduct the experiments of the article Dynamic Ranking and Translation Synchornization https://arxiv.org/pdf/2207.01455.pdf
#
# This module contains functions used to construct particular graphs, union graphs and check for connectivity

import numpy as np
import scipy
import networkx as nx

### Creation of incidence matrices ###

def list_edges(M):
    '''
    Get the list of pairs of indexes (i,j), i<j, such that (i,j) is an edge in the graph
    Input :
        M : adjacency matrix of a graph
    Output :
        list of pairs
    '''
    non_zero = np.transpose(np.nonzero(np.triu(M)))
    return list(non_zero)
    
def incidence(M):
    '''
    Create the sparse incidence matrix of a graph from the adjacency matrix
    Inout :
        M : adjacency matric of a graph
    Output : 
        Sparse Incidence matrix Q where (i,j) an edge,i<j, implies that Q[i,j] = -1 and Q[j,i] = 1
    '''
    n = np.shape(M)[0]
    edges = list_edges(M)
    Q = np.zeros((n,len(edges)))
    for i in range(len(edges)):
        a,b = edges[i]
        Q[min(a,b),i] = -1
        Q[max(a,b),i] = 1
    return scipy.sparse.csr_matrix(Q)

def diag_incidence(A):
    '''
    Create a sparse block diagonal matrix from an array of incidence matrices
    Input :
        A: 3 dimensional array such that A[i,:,:] is the adjacency matrix of the i-th graph
    Output :
        Sparse block diagonal matrix with incidence matrix on the diagonal
    '''
    diag = []
    for t in range(np.shape(A)[0]):
        diag.append(incidence(A[t,:,:]))
    return scipy.sparse.block_diag(diag)

### Incidence matrix of special graphs ###

def incidence_Kn(n):
    '''
    Create incidence matrix of the complete graph on n vertices
    '''
    A = np.ones((n,n))-np.eye(n)
    return incidence(A)

def incidence_path(n):
    '''
    Create incidence matrix of the path graph on n vertices
    '''
    A = np.diag(np.ones(n-1),1)+np.diag(np.ones(n-1),-1)
    return incidence(A)

#### Union graph on a neighborhood ####

def neighborhood(t,delta,T):
    '''
    Create an array of length T+1 with boolean True if grid[i] belongs to the neighborhood of t
    The neighborhood of t is composed of the points k/T such that |t-k/T| <= delta/T.
    Input:
        t: time in [0,1] around which we create the neighborhood
        delta: size-parameter
        T: grid of size T+1
    Output:
        T+1-array of booleans
    '''
    grid = np.arange(0,1+1/(2*T),1/T)
    return abs(grid-t)<= delta/T

def union_graph(A,delta,t):
    '''
    Create the adjacency matrix of the union graph G_delta(t), each coefficient is equal to the number of graphs in the neighborhood involved in the edge (i,j).
    Remark : To get the union of all the inputs, use union_graph(A,T,0.5)
    Input:
        A: (T+1)-N-N array containing the adjacency matrix of the observed graphs.
        t: time in [0,1] around which we create the union graph
        delta: size-parameter
    Output:
        N-N array containing the adjacency matrix of the union graph
    '''
    T,N = np.shape(A)[0:2]
    N_delta = neighborhood(t,delta,T-1)
    A_delta = sum(A[N_delta,:,:])
    return A_delta

def graph_proba(graph_matrix):
    '''
    Estimate the probability of a G(n,p) graph from its adjacency matrix
    '''
    N = np.shape(graph_matrix)[0]
    return 2*np.sum(np.triu(graph_matrix,k=1))/(N*(N-1))

#### Check fo connectivity or completeness ###

def connected(A):
    G = nx.from_numpy_matrix(A)
    return nx.is_connected(G)

print('graph')

