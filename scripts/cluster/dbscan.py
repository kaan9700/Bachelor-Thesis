import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import sys
sys.path.insert(0, "..")
from functions import load_np_array_pickle
import pickle

def dbscan_cluster():
    # Abstandsmatrix
    cross_correlations = load_np_array_pickle('../../files/correlation_files/correlation_epochs.pickle')
    # Convert cross-cluster matrix to distance matrix
    dist_matrix = 1 - cross_correlations
    best_score = -1
    best_eps = 0
    best_min_samples = 0

    for eps in np.arange(0.1, 1.0, 0.1):
        print(eps)
        for min_samples in range(1, dist_matrix.shape[0]):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
            labels = dbscan.fit_predict(dist_matrix)

            # Check if number of clusters is greater than 1
            if len(set(labels)) > 1:
                score = silhouette_score(dist_matrix, labels, metric='precomputed')
                if score > best_score:
                    best_score = score
                    best_eps = eps
                    best_min_samples = min_samples

    # Re-run DBSCAN with best parameters
    dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples, metric='precomputed')
    labels = dbscan.fit_predict(dist_matrix)
    print("Optimal DBSCAN parameters: eps = {}, min_samples = {}".format(best_eps, best_min_samples))
    print("Cluster labels:", labels)
    with open("../../files/cluster_files/dbscan_epochs.pickle", 'wb') as f:
        pickle.dump(labels, f)

if __name__ == "__main__":
    dbscan_cluster()
