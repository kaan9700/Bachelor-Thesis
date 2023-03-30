import sys
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score
import pickle5 as pickle
sys.path.insert(0, "..")
from functions import load_np_array_pickle


def affinity_cluster():
    cross_correlations = load_np_array_pickle('../../epochs_files/correlation_compare.pickle')

    best_score = -1
    best_damping = 0

    for damping in np.arange(0.5, 1.0, 0.05):
        ap = AffinityPropagation(damping=damping)
        labels = ap.fit_predict(cross_correlations)

        # Check if number of clusters is greater than 1
        if len(set(labels)) > 1:
            score = silhouette_score(cross_correlations, labels)
            if score > best_score:
                best_score = score
                best_damping = damping

    # Re-run Affinity Propagation with best parameters
    ap = AffinityPropagation(damping=best_damping)
    labels = ap.fit_predict(cross_correlations)
    print("Optimal Affinity Propagation parameters: damping = {}".format(best_damping))
    print("Cluster labels:", labels)
    with open("../../files/cluster_files/ap.pickle", 'wb') as f:
        pickle.dump(labels, f)


if __name__ == "__main__":
    affinity_cluster()