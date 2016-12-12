''' Module to Implement Linear Regression '''

import numpy as np
import matplotlib.pyplot as plt


def data_processing(filename):
    ''' Take a file in the form temp, energy and return it as an array of arrays '''

    # Initialise the sample space and label arrays
    labeled_sample_space = []

    for line in open(filename):
        temp, energy = line.strip().split(" ")
        labeled_sample_space.append([float(temp), float(energy)])

    return labeled_sample_space



def linear_regression(sample_space):
    '''Implement the linear regression model as defined in the lecture'''

    # Set y = transpose of the last column of the sample space
    prediction = np.matrix([observation[-1] for observation in sample_space])
    prediction = np.transpose(prediction)
    # Get the matrix X from the sample_space
    samples = [observation[:-1] for observation in sample_space]
    # Append '1' to each row.
    for row in samples:
        row.append(1)

    # Covert to an numpy matrix
    samples = np.matrix(samples)

    # Find the transpose and the inverse of X^T*X
    sample_transpose = np.transpose(samples)
    inverse = np.matrix.getI(sample_transpose*samples)

    # Put it all together to get w = (X^TX)^{-1}y
    optimal_parms = (inverse*sample_transpose)*prediction

    return optimal_parms
