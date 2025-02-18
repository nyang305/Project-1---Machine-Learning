from datapreprocessor import DataPreprocessor
from distance_voting import minkowski_distance, plurality_vote
import numpy as np
import pandas as pd
import random

"""
    
Null models are naive algorithms that will serve as a comparison baseline
for k-nearest neighbour, nonparametric pipeline.
    
"""

def null_classifier(y_train):
    
    """
    
    Null classifier returns the most common class label for classification tasks.

    """
    
    values, counts = np.unique(y_train, return_counts = True)
    return values[np.argmax(counts)]

def null_regressor(y_train):
    
    """
    
    Null regressor returns the average of the outputs for regression tasks.

    """
    
    return np.mean(y_train)

def run_null_model_classification(X_train, y_train, X_test, y_test):
    
    """
    
    X_train, y_train, X_test, y_test are numpy arrays, 
    but we only need y_train and y_test for the null model.
    Returns accuracy of the null classifier on the test set.
    
    """
    
    # Train the null classifier to get the most common class
    majority_class = null_classifier(y_train)
    # Predict that class for every example in X_test
    predictions = [majority_class] * len(y_test)
    # Compute accuracy
    accuracy = np.mean(predictions == y_test)
    return accuracy

def run_null_model_regression(X_train, y_train, X_test, y_test):
    
    """
    
    Returns MSE of the null regressor on the test set.
    
    """
    
    # Train the null regressor (the mean of the training targets)
    mean_val = null_regressor(y_train)
    # Predict that mean for every example in y_test
    predictions = [mean_val] * len(y_test)
    # Compute mean square error
    mse = np.mean((np.array(predictions) - y_test)**2)
    return mse