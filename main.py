import numpy as np
from sklearn import preprocessing, cluster
import networkx as nx
import matplotlib.pyplot as plt


def getData():
    edgelist = np.genfromtxt("example1.dat", delimiter=",", dtype=int)
    graph = nx.Graph()
    for edge in edgelist:
        graph.add_edge(edge[0], edge[1])

    adjacency = nx.adjacency_matrix(graph)

    adjacency = adjacency.todense()
    nx.draw(graph, with_labels=True)
    plt.show()
    return adjacency, graph

def spectralClustering(adjacency_matrix):
    sigma = 5
    k = 4 #choose num clusters for KMeans
    num_eigenvectors = 5
    affinity = np.zeros(adjacency_matrix.shape)
    sigma_square = sigma**2
    for i in range(adjacency_matrix.shape[0]): ##FORM AFFINITY MATRIX
        for j in range(adjacency_matrix.shape[0]):
            if i == j:
                affinity[i, j] = 0
            else:
                affinity[i, j] = np.exp(-np.linalg.norm(adjacency_matrix[:, i]-adjacency_matrix[:, j])/(2*sigma_square))
    row_sums = np.sum(affinity, axis=1)
    row_sums = row_sums**(-1/2) #Sum rows, etc
    d = np.diag(row_sums) ##DIAGONAL MATRIX FROM SUM OF ROWS
    laplace = np.dot(d, affinity).dot(d)#form L matrix
    eigval, eigvec = np.linalg.eig(laplace) #COMPUTE EIGENVECTORS #egenvektorer egenvärden
    kmeans = cluster.KMeans(n_clusters=k)
    x = eigvec[:, :num_eigenvectors] #choose the greatest eigenvectors
    y = preprocessing.normalize(x, axis=1)  # normalize
    labels = kmeans.fit_predict(y) #K Means clustering
    nodes_by_label = {}
    for label in np.unique(labels):
        nodes_by_label[label] = np.where(labels == label)
    for key in nodes_by_label.keys():
        print(key, " : ", nodes_by_label[key])
    return labels

def clusteringFromAdjacency(adjacency_matrix):
    k = 4
    kmeans = cluster.KMeans(n_clusters=k)
    sum = np.sum(adjacency_matrix, axis=0)
    sum_squared = 1/np.sqrt(sum)
    diagonal = np.diag(sum_squared)
    l = np.dot(diagonal, adjacency_matrix).dot(diagonal)
    eigval2, eigvec2 = np.linalg.eig(l)
    x2 = eigvec2[:, :6]
    y2 = preprocessing.normalize(x2, axis=1)
    labels2 = kmeans.fit_predict(y2)
    return labels2


adjacent, graph = getData()
labels = spectralClustering(adjacent)
#labels2 = clusteringFromAdjacency(adjacenct)
nx.draw_networkx(graph, with_labels=True, node_color=labels, cmap="Dark2") #Nodes are colorcoded according to cluster label
#nx.draw_networkx(graph, with_labels=True, node_color=labels2, cmap="Dark2")
plt.title("Using affinity matrix")
plt.show()


