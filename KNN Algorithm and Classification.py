''' Module to answer Machine Learning Assignment 2, Question 1 '''

from collections import Counter
from scipy.spatial import KDTree
import matplotlib.markers as mmarkers
import matplotlib.pyplot as plt
import random
import numpy as np



def data_processing(filename):
    ''' Take a file in the form length(mm), width(cm) and flower type (0, 1, or 2) and return an array of points (length (mm), width (mm)) and an array of flower types. '''

    # Initialise the sample space and label arrays
    labeled_sample_space = []

    for line in open(filename):
        length, width, label = line.strip().split(" ")
        labeled_sample_space.append([[float(length), round(float(width)*10, 2)], label])

    # Data returned in this manner to make cross-validation easier.
    return labeled_sample_space

def k_nearest_neighbours(kay, labeled_sample_space, test_data):
    ''' For a given sample_space and labels array, carries out the K-NN algorithm'''

    sample_space = [elem[0] for elem in labeled_sample_space]
    labels = [elem[1] for elem in labeled_sample_space]


    # Create a 2d spatial binary search tree for the points in the training data
    search_tree = KDTree(sample_space)

    # For each point in the test data, traverse the KDTree in order to find the 
    # k nearest neighbours.  Return only the positions of the neighbours in 
    # search_tree.data
    neighbours = search_tree.query(test_data, k=kay)[1]


    flower_type_results = []

    # When k == 1, there is no need for frequency checks, as there is only one
    # result.  Furthermore, the data structure of neighbours is 1-d, rather than 2-d
    if kay == 1:
        for sample in neighbours:
            # Look up the position of the nearest neighbour in the labels list, to find the corresponding label
            flower_type_results.append(labels[sample])
    else:
        for sample in neighbours:
            # Get list of the flower types for the nearest neighbours
            flower_types = [labels[index] for index in sample]
            # Find the label for the most common type.
            # Note: the counter object orders elements with equal counts arbitrarily
            variety = Counter(flower_types).most_common(1)[0][0]
            flower_type_results.append(variety)

    return flower_type_results

def cross_validation(kay, labeled_sample_space):
    '''Implement 5-fold cross validation on the given data and the K-NN algorithm'''

    ## Used later to collect error statistics across cross validation folds
    errors = []
    # For each fifth of the labelled sample_space, carry out the K-NN algorithm
    for index in range(0, 5):
        # Create the test and training data arrays
        test = labeled_sample_space[index::5]
        training = [elem for count, elem in enumerate(labeled_sample_space) if count%5 != index]

        test_unlabelled = [elem[0] for elem in test]
        # Carry out the K-NN algorithm
        flower_results = k_nearest_neighbours(kay, training, test_unlabelled)

        # Calculate the cross validation loss (0-1 loss used)
        classification_error = 0.0
        ## Retrieve the correct labels
        test_labels = [elem[1] for elem in test]
        for variety in range(0, len(test_labels)):
            ## Test against the K-NN derived labels
            if test_labels[variety] != flower_results[variety]:
                classification_error += 1
        errors.append(classification_error/len(test_labels))
        

    # Average the cross validation loss over the 5 folds
    aver_error = np.mean(errors)

    return aver_error

def get_k_best(labeled_sample_space, runtimes):
    ''' Run the cross validation run times times for each value of k.
    Then return the average loss.'''

    # Randomly permute the labeled_sample_space

    loss = [0 for num in range(1, 26) if num%2 != 0]

    # Repeat the process several times to help remove the influence of randomness.
    for _ in range(0, runtimes):
        random.shuffle(labeled_sample_space)  
        for index, kay in enumerate([num for num in range(1, 26) if num%2 != 0]):
            loss[index] += cross_validation(kay, labeled_sample_space)/runtimes

    # Take the average loss from each run above, and return it.
    kays = [num for num in range(1, 26) if num%2 != 0]

    # Find the value of k that minimises loss.
    best_k_index = loss.index(min(loss))
    best_k = kays[best_k_index]

    return loss, kays, best_k

def find_average_error(known, test):
    '''Take the test results from the algorithm and return the 0-1 loss'''
    # Reset the classification error count
    errors = 0
    # Compare against each other (test data).
    for elem in range(0, len(test)):
        ## Test against the K-NN derived labels
        if test[elem] != known[elem]:
            errors += 1

    return errors/len(test)
