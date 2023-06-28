import numpy as np
import scipy
import networkx as nx

### Creation of incidence matrices ###

def list_edges(M):
    '''
    M : adjacency matrix of a graph
    Get the list of pairs of indexes (i,j), i<j, such that (i,j) is an edge in the graph
    '''
    non_zero = np.transpose(np.nonzero(np.triu(M)))
    return list(non_zero)
    
def incidence(M):
    '''
    M : adjacency matric of a graph
    Output : Sparse Incidence matrix Q where (i,j) an edge,i<j, implies that Q[i,j] = -1 and Q[j,i] = 1
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
    Create a sparse block diagonal matrix from a list of incidence matrices
    '''
    diag = []
    for t in range(np.shape(A)[0]):
        diag.append(incidence(A[t,:,:]))
    return scipy.sparse.block_diag(diag)

### Incidence matrix of special graphs ###

def incidence_Kn(n):
    A = np.ones((n,n))-np.eye(n)
    return incidence(A)

def incidence_path(n):
    A = np.diag(np.ones(n-1),1)+np.diag(np.ones(n-1),-1)
    return incidence(A)

#### Union graph on a neighborhood ####

def neighborhood(t,delta,T):
    '''
    get an array of length T+1 with boolean True if grid[i] belongs to the neighborhood of t
    The neighborhood of t is composed of the points k/T such that |t-k/T| <= delta/T.
    '''
    grid = np.arange(0,1+1/(2*T),1/T)
    return abs(grid-t)<= delta/T

def union_graph(A,delta,t):
    '''
    get the adjacency matrix of the union graph G_delta(t), each coefficient is equal to the number of graphs in the neighborhood involved in the edge (i,j).
    A: (T+1)-N-N array containing the adjacency matrix of the observed graphs.
    Remark : To get the union of all the inputs, use union_graph(A,T,0.5)
    '''
    T,N = np.shape(A)[0:2]
    N_delta = neighborhood(t,delta,T-1)
    A_delta = sum(A[N_delta,:,:])
    return A_delta

def graph_proba(graph_matrix):
    N = np.shape(graph_matrix)[0]
    return 2*np.sum(np.triu(graph_matrix,k=1))/(N*(N-1))

#### Check fo connectivity or completeness ###

def connected(A):
    G = nx.from_numpy_matrix(A)
    return nx.is_connected(G)

print('graph')

