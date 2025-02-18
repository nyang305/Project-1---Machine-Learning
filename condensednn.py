from datapreprocessor import DataPreprocessor
from k_nearest import knn_classify
from distance_voting import minkowski_distance
import numpy as np
import pandas as pd
import random

def condensed_nearest_neighbor (X_train, y_train, p = 2):
    
    """
    
    Implements the condensed nearest neighbor algorithm.
    Starts with one random example and iteratively adds misclassified points.
    Uses 1-NN for prediction.
    
    Returns:
    X_condensed features the input data remaining in the condensed set
    y_condensed contains corresponding class labels for remaining data
    
    """
    
    n = X_train.shape[0]
    indices = list(range(n))
    condensed_indices = [random.choice(indices)]
    changed = True
    while changed:
        changed = False
        for i in indices:
            if i in condensed_indices:
                continue
            x_test = X_train[i]
            y_pred = knn_classify (X_train[condensed_indices], y_train[condensed_indices], x_test, k = 1, p = p)
            if y_pred != y_train[i]:
                condensed_indices.append(i)
                changed = True
    return X_train[condensed_indices], y_train[condensed_indices]