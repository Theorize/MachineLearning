
''' Module to implement K-Means Clustering '''

import numpy as np
from scipy.spatial import distance


def data_processing(filename):
    ''' Take a csv file in the with columns representing the sample space point and lastly it's label and return an array of points and an array of labels. '''

    # Initialise the sample space and label arrays
    sample_space = []
    labels = []

    for line in open(filename):
        # Get the data on each line, strip extra characters and separate by comma
        data = line.strip().split(",")

        # Append all but the last entry to the sample space record
        sample_space.append([float(measure) for measure in data[:-1]])
        # Append label to label array
        labels.append(float(data[-1]))

    # Data returned in this manner to make cross-validation easier.
    return np.array(sample_space), labels



def k_means_initialisation(sample_space, k):
    ''' Initialise the k means clustering algorithm before passing the data onto the iterative step. '''

    # Create a clustering array and the centroid array.  The clustering array will record the cluster number for each point in the sample space.
    clusterings = [0 for _ in  range(0, len(sample_space))]
    centroid_array = sample_space[:k]

    # Place the centroid array elements in the corresponding clusters
    for index in range(0, k):
        clusterings[index] = index

    # Pass the data into iteration.
    return k_means_clustering(sample_space, clusterings, centroid_array, k, 0)



def k_means_clustering(sample_space, clusterings, centroid_array, k, count):
    ''' Given a current clustering, perform one iteration of the k_means clustering algorithm and then pass the clustering on for further iterations if neccessary. '''

    # For each element in the sample_space, set it to below to the cluster with the closest centroid.
    new_clusterings = []
    for element in sample_space:
        new_clusterings.append(closest_centroid(element, centroid_array, k))
    count += 1

    # Check if the clustering array has changed.  If so, perform a recursive iteration.  If not, return this clustering as final.
    if np.array_equal(clusterings, new_clusterings):
        print("The total number of interations performed was", count)
        return new_clusterings
    else:
        # Calculate new centroids
        new_centroid_array = new_centroids(sample_space, new_clusterings, k)
        return k_means_clustering(sample_space, new_clusterings, new_centroid_array, k, count)



def closest_centroid(point, centroid_array, k):
    ''' Return the number of the cluster for which the centroid is the closest to the given point'''
    distance_to_cluster = distance.euclidean(point, centroid_array[0])
    closest_cluster = 0
    for index in range(1, k):
        if distance.euclidean(point, centroid_array[index]) < distance_to_cluster:
            # Then index is the closer cluster, so record it as such
            closest_cluster = index
            distance_to_cluster = distance.euclidean(point, centroid_array[index])
    return closest_cluster



def new_centroids(sample_space, clusterings, k):
    ''' Calculate the new centroid to be the mean of the members of its cluster'''

    cluster_means = [0 for _ in range(0, k)]
    cluster_counts = [0 for _ in range(0, k)]
    centroid_array = [0 for _ in range(0, k)]
    for index, value in enumerate(sample_space):
        cluster = int(clusterings[index])

        cluster_means[cluster] = np.add(cluster_means[cluster], value)
        cluster_counts[cluster] += 1
    for index in range(0, k):
        centroid_array[index] = cluster_means[index]/cluster_counts[index]

    return centroid_array

