from datapreprocessor import DataPreprocessor
from distance_voting import minkowski_distance, plurality_vote
import numpy as np
import pandas as pd
import random

def knn_classify(X_train, y_train, x_test, k=3, p=2):
    
    """
    
    k-nearest neighbour classifier that predicts the class of 
    a single test instance.
    
    """
    
    distances = []
    for i in range(X_train.shape[0]):
        dist = minkowski_distance(X_train[i], x_test, p)
        distances.append((dist, y_train[i]))
    distances.sort(key=lambda tup: tup[0])
    neighbours = [label for (_, label) in distances[:k]]
    return plurality_vote(neighbours)

def knn_regress(X_train, y_train, x_test, k=3, p=2, gamma=1.0):
    
    """
    
    k-nearest nieghbour regressor that predicts 
    a numeric target value using a Gaussian (RBF) kernel.
    
    """
    distances = []
    for i in range(X_train.shape[0]):
        dist = minkowski_distance(X_train[i], x_test, p)
        distances.append((dist, y_train[i]))
    distances.sort(key=lambda tup: tup[0])
    neighbours = distances[:k]
    weights = np.array([np.exp(-gamma * (d**2)) for d, _ in neighbours])
    neighbour_values = np.array([val for _, val in neighbours])
    if weights.sum() == 0: # Means neighbours are really far away
        return neighbour_values.mean() # Return mean to neighbour values to avoid dividing by 0
    return np.dot(weights, neighbour_values) / weights.sum()