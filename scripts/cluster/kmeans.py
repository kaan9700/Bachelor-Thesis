

import sys
sys.path.insert(0, "..")
from functions import load_np_array_pickle
import pickle
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt



# Find optimal number of clusters using Gap-Statistics method
def opt_num(X):
    """
    This function calculates and returns the optimal number of clusters
    using the gap statistic method.
    """
    # Initializations
    n_clusters = range(1, 10)
    gaps = np.zeros(len(n_clusters))
    stds = np.zeros(len(n_clusters))

    # Compute the gap statistic for each value of k
    for i, k in enumerate(n_clusters):
        # Fit the k-means model
        km = KMeans(n_clusters=k, n_init=10)
        print(i)
        km.fit(X)

        # Calculate within-cluster dispersion
        disp = km.inertia_

        # Generate reference data set and calculate its dispersion
        ref_disp = np.zeros(10)
        for j in range(10):
            print(j)
            # Generate random data set
            random_data = np.random.random_sample(size=X.shape)

            # Fit k-means to random data set
            km = KMeans(n_clusters=k, n_init=10)

            km.fit(random_data)

            # Calculate within-cluster dispersion
            ref_disp[j] = km.inertia_

        # Calculate gap statistic and standard deviation
        gaps[i] = np.mean(np.log(ref_disp)) - np.log(disp)
        stds[i] = np.sqrt(1 + 1 / 10) * np.std(np.log(ref_disp))

    # Calculate the optimal number of clusters
    opt_k = gaps.argmax() + 1

    # Plot the gap statistic
    plt.errorbar(n_clusters, gaps, yerr=stds, fmt='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Gap statistic')
    plt.title('Gap statistic vs. number of clusters')
    plt.show()

    return opt_k



def kmeans_cluster():
    # Abstandsmatrix
    cross_correlations = load_np_array_pickle('../../files/correlation_files/correlation_windows.pickle')
    print(cross_correlations.shape)
    distance_matrix = 1 - cross_correlations
    # Call the function and print the optimal number of clusters
    opt_k = opt_num(distance_matrix)
    print("Optimal number of clusters:", opt_k)
    # Berechne optimale Anzahl an Clustern mithilfe des Elbow-Methods
    sum_of_squared_distances = []
    K = range(1, 10)
    for k in K:
        print(k)
        km = KMeans(n_clusters=k, n_init=10)
        km = km.fit(distance_matrix)
        sum_of_squared_distances.append(km.inertia_)

    plt.plot(K, sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Summe der quadratischen Abstände')
    plt.title('Elbow Method zur Bestimmung der optimalen Anzahl an Clustern')
    plt.show()

    # Anwenden des k-Means-Algorithmus auf die Abstandsmatrix
    optimal_number_of_clusters = int(input("Wähle die optimale Anzahl an Clustern aus dem Elbow-Plot aus: "))
    kmeans = KMeans(n_clusters=optimal_number_of_clusters, n_init=10)

    kmeans.fit(distance_matrix)
    labels = kmeans.labels_
    print("Cluster-Zuordnungen: ", labels)
    print(len(labels))
    with open("../../files/cluster_files/kmeans.pickle", 'wb') as f:
        pickle.dump(labels, f)

if __name__ == "__main__":
    kmeans_cluster()