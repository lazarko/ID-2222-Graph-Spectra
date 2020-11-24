import numpy as np
import sklearn as sk
import scipy.sparse as sps


def getData():
    edgelist = np.genfromtxt("example1.dat", delimiter=",", dtype=int)
    print(edgelist.dtype)

    if edgelist.shape[1] == 3:
        weight = edgelist[:, 2]
        edgelist = edgelist[:, 0:2]
    else:
        weight = np.ones((len(edgelist), 1))
    i, j = edgelist[:, 0], edgelist[:, 1]
    #print(i)
    #print(edgelist)
    dim = max(max(set(i), set(j)))
    test = min(min(set(i), set(i)))
    #print(test)
    #print(dim)
    adjacency_matrix = np.zeros((dim, dim))
    #print(adjacency_matrix.shape)
    #print(adjacency_matrix)

    #matrix = sps.lil_matrix((dim, dim))
    #print("size", adjacency_matrix)
    for i, j, w in zip(i, j, weight):
        adjacency_matrix[i-1, j-1] = w #nodes are numbered from 1, not 0

    #print(adjacency_matrix)
    return adjacency_matrix

def spectralClustering(adjacency_matrix):
    sigma = 1 #WHAT IS SIGMA?
    k = 6 #WHAT NUMBER OF KLUSTERS??
    affinity = np.zeros(adjacency_matrix.shape)
    for i in range(adjacency_matrix.shape[0]): ##FORM AFFINITY MATRIX
        for j in range(adjacency_matrix.shape[0]):
            affinity[i,j] = np.exp(((adjacency_matrix[:,i] - adjacency_matrix[:,j])**2)/(2*sigma**2))
    row_sums = np.sum(affinity, axis=1)
    d = np.diag(row_sums) ##DIAGONAL MATRIX FROM SUM OF ROWS
    laplacian = d**(-1/2)*affinity*d**(-1/2) #FORM L MATRIX
    eigval, eigvec = np.linalg.eig(laplacian) #COMPUTE EIGENVECTORS
    eigvec = eigvec.real #sometimes due to floating point errors in numpy ou get complex eigenvectors
    x = eigvec[:, 0:k]
    y = sk.preprocessing.normalize(x, axis=1)
    kmeans = sk.cluster.KMeans(n_clusters=k)
    predicted = kmeans.fit_predict(y)
    #assign original node s[i] (represented by its vector of edges) to cluster j if row i in Y is assigned to cluster j





adjacent = getData()
spectralClustering(adjacent)
