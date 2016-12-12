
''' Module to answer Machine Learning Assignment 3, Question 3 '''

import numpy as np

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




def perform_pca(sample_space, k):
    '''Perform the pca on a given sample space, and return (ordered by decreasing e-vals):
        - e-vals
        - e-vecs
        - cumulative explained variance
        - projection of sample space by the first k components.
    '''

    empirical_mean = np.mean(sample_space, axis=0)
    empirical_covariance = np.cov(sample_space, rowvar=0)
    # Now get the eigenvalues and eigenevectors
    w_evals, v_evecs = np.linalg.eig(empirical_covariance)
    # Note that the documentation https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linalg.eig.html states that: "the colunmn v[:,i] is the eigenvector corresponding to the eigenvalue w[i]."

    # Get the order of the eigenvalues so that they're in decreasing order
    ordering = np.abs(w_evals).argsort()[::-1]
    # Reorder the eigenvalues and eigenvectors in this order
    w_evals = w_evals[ordering]
    # evecs are transposed first so that the index v_evecs[item] corresponds to an evec
    v_evecs = np.transpose(v_evecs)
    v_evecs = v_evecs[ordering]

    # Calculate the cummulative explained variance for each e-val
    cumulative_explained_var = np.cumsum(w_evals)/np.sum(w_evals)

    # Compute a matrix composed of the first k eigenvectors of S
    # Note that transposition occurred earlier, so should not be repeated here.
    projection_matrix = v_evecs[:k]
    # Calculate the projection of sample space by the first k components.
    projected_sample_space = np.dot(np.real(projection_matrix),
                                    np.transpose(sample_space-empirical_mean))

    # Return analysis
    return w_evals, v_evecs, cumulative_explained_var, projected_sample_space
    