import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "..")
from functions import load_np_array_pickle
import pickle
from scipy.spatial.distance import squareform

def hierachical_cluster():
    # Abstandsmatrix
    cross_correlations = load_np_array_pickle('../../files/correlation_files/correlation_windows.pickle')
    print(len(cross_correlations[0]))
    distance_matrix = 1 - cross_correlations
    condensed_distance_matrix = squareform(distance_matrix)
    # Berechne den Hierarchischen Clustering
    Z = sch.linkage(condensed_distance_matrix, method='ward')

    # Plotte den Dendrogramm
    plt.title("Agglomerativer Hierarchischer Clustering")
    sch.dendrogram(Z, labels=np.arange(len(distance_matrix)))
    plt.show()

    # Bestimme die ideale Anzahl an Clustern
    last = Z[-10:, 2]
    last_rev = last[::-1]
    idxs = np.arange(1, len(last) + 1)
    plt.plot(idxs, last_rev)

    acceleration = np.diff(last, 2)  # 2nd Derivative of the distances
    acceleration_rev = acceleration[::-1]
    plt.plot(idxs[:-2] + 1, acceleration_rev)
    plt.show()
    k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
    print("Optimale Anzahl an Clustern: ", k)

    # Hierarchischer Cluster Algorithmus
    linked = linkage(condensed_distance_matrix, method='ward')
    # Clusterberechnung
    clusters = fcluster(linked, k, criterion='maxclust')
    print("Cluster-Zuordnung für jeden Datenpunkt:  \n", clusters)
    with open("../../files/cluster_files/hierachical_windows.pickle", 'wb') as f:
         pickle.dump(clusters, f)

if __name__ == "__main__":
    hierachical_cluster()
