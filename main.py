import numpy as np
from sklearn import preprocessing, cluster
import scipy.sparse as sps
import networkx as nx
import scipy as sp


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

    graph = nx.Graph()
    for edge in edgelist:
        graph.add_edge(edge[0], edge[1])
    print(graph.number_of_nodes())
    adjacency = nx.adjacency_matrix(graph)
    print(adjacency.todense())
    print(adjacency_matrix)
    adjacency = adjacency.todense()
    return adjacency, graph

def spectralClustering(adjacency_matrix):
    sigma = 2 #WHAT IS SIGMA?
    k = 6 #WHAT NUMBER OF KLUSTERS??
    affinity = np.zeros(adjacency_matrix.shape)
    sigma_square = sigma**2
    for i in range(adjacency_matrix.shape[0]): ##FORM AFFINITY MATRIX
        for j in range(adjacency_matrix.shape[0]):
            if i == j:
                affinity[i, j] = 0
            else:
                affinity[i, j] = np.exp(-np.linalg.norm(adjacency_matrix[:, i]-adjacency_matrix[:, j])/(2*sigma_square))
    row_sums = np.sum(affinity, axis=1)
    row_sums = row_sums**(-1/2)
    d = np.diag(row_sums) ##DIAGONAL MATRIX FROM SUM OF ROWS
    new_d = d**-0.5
    laplace = np.dot(d, affinity).dot(d)#form L matrix
    #laplacian = d**(-1/2)*affinity*d**(-1/2) #FORM L MATRIX
    eigval, eigvec = np.linalg.eig(laplace) #COMPUTE EIGENVECTORS
    eigvec = eigvec.real #sometimes due to floating point errors in numpy ou get complex eigenvectors
    sorted = np.sort(eigvec)[:, ::-1]

    x = sorted[:, 0:k]
    y = preprocessing.normalize(x, axis=1)
    kmeans = cluster.KMeans(n_clusters=k)
    labels = kmeans.fit_predict(y)
    classification = []
    for i, label in enumerate(labels):
        print(i, " -->", label)

    # TODO om rad i av Y hamnar i cluster j, lägg nod i i cluster j
    return labels
    #assign original node s[i] (represented by its vector of edges) to cluster j if row i in Y is assigned to cluster j





adjacent, graph = getData()
labels = spectralClustering(adjacent)
nx.draw_networkx(graph, node_color=labels, cmap="Dark2")
