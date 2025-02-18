from datapreprocessor import DataPreprocessor
import numpy as np
import pandas as pd
import random

def stratified_5x2_cv (df, target_column):
    
    """
    
    Generates 5 by 2 cross-validation splits for classification.
    Stratifies by class label so each half of the data is roughly
    the same class distribution.
    
    """
    
    folds = []
    classes  = df[target_column].unique()
    class_dfs = {c: df[df[target_column] == c] for c in classes}
    
    for _ in range(5):
        train_list = []
        test_list = []
        for c, subdf in class_dfs.items():
            subdf = subdf.sample(frac = 1, random_state = random.randint (0, 10000)).reset_index(drop = True)
            half = len(subdf) // 2
            train_list.append(subdf.iloc[:half])
            test_list.append(subdf.iloc[half:])
        fold1 = pd.concat(train_list).sample(frac = 1).reset_index(drop = True)
        fold2 = pd.concat(test_list).sample(frac = 1).reset_index(drop = True)
        folds.append((fold1, fold2))
        folds.append((fold2, fold1))
    return folds

def simple_5x2_cv(df, target_column):
    
    """
    
    For regression tasks or if stratification is not needed. 
    Simply shuffles the data and splits it in half each time.
    
    """
    
    folds = []
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    half = len(df_shuffled) // 2
    for _ in range(5):
        train1 = df_shuffled.iloc[:half]
        test1 = df_shuffled.iloc[half:] # Assigns second half to test1 as testing set
        folds.append((train1, test1)) # Appends tuple to folds
        folds.append((test1, train1)) # Appends reversed tuple to folds
        df_shuffled = df_shuffled.sample(frac=1).reset_index(drop=True) # Shuffles df for next iteration
    return folds