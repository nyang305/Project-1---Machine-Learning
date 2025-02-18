from datapreprocessor import DataPreprocessor
import numpy as np
import pandas as pd
import random

"""

The following provides the distance and voting functions

"""

def minkowski_distance (x, y, p = 2):
    
    #Euclidean distance
    
    return np.sum(np.abs(x - y) ** p) ** (1.0 / p)

def plurality_vote (neighbours):
    
    """
    
    Given a list of class labels, the most common label will be returned.
    If a tie occurs, a label will be randomly chosen.

    """
    
    counts = {}
    for label in neighbours:
        counts[label] = counts.get(label, 0) + 1
    max_count = max(counts.values())
    candidates = [label for label, count in counts.items() if count == max_count]
    return random.choice(candidates)