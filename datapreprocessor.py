import pandas as pd 
import numpy as np 
import random
from copy import deepcopy

# A dictionary mapping certain known dataset filenames to their columns
# This became necessary due to some data files not containing column headers
DATASET_COLUMNS = {
    "abalone.data": [
        "Sex", 
        "Length", 
        "Diameter", 
        "Height", 
        "Whole weight",
        "Shucked weight", 
        "Viscera weight", 
        "Shell weight", 
        "Rings"
    ],
    "breast-cancer-wisconsin.data": [
        "Sample_code_number", 
        "Clump_Thickness", 
        "Uniformity_Cell_Size",
        "Uniformity_Cell_Shape", 
        "Marginal_Adhesion",
        "Single_Epithelial_Cell_Size", 
        "Bare_Nuclei", 
        "Bland_Chromatin",
        "Normal_Nucleoli", 
        "Mitoses", 
        "Class"
    ],
    "car.data": [
        "buying",
        "maint",
        "doors",
        "persons",
        "lug_boot",
        "safety",
        "class"
    ],
    "machine.data": [
        "vendor name",
        "Model Name",
        "MYCT",
        "MMIN",
        "MMAX",
        "CACH",
        "CHMIN",
        "CHMAX",
        "PRP",
        "ERP"
    ],
    "house-votes-84.data": [
        "Class Name",
        "handicapped-infants",
        "water-project-cost-sharing",
        "adoption-of-the-budget-resolution",
        "physician-fee-freeze",
        "el-salvador-aid",
        "religious-groups-in-school",
        "anti-satellie-test-ban",
        "aid-to-nicaraguan-contras",
        "mx-missile",
        "immigration",
        "synfuels-corporation-cutback",
        "education-spending",
        "superfund-right-to-sue",
        "crime",
        "duty-free-exports",
        "export-administration-act-south-africa"
    ]
}

class DataPreprocessor:
    def __init__(self, numeric_cols = None, categorical_cols = None):
        
        """
        
        This class processees the data.
        numeric_cols: columns that store numeric data
        categorical_cols: columns that store categorical data
            
        """
        
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.numeric_means = {}
        self.numeric_stds = {}
        self.cat_levels = {} 
        
    def load_data(self, filepath, target_column, drop_columns = None):
        
        """
        
        Loads a CSV file into a pandas DataFrame. 
        If the filename is in DATASET_COLUMNS, that means we know it has no header
        and we want to supply them ourselves. Otherwise, we'll just read normally.
        
        """
        
        import os
        filename = os.path.basename(filepath).lower()
        
        if filename in DATASET_COLUMNS:
            # Determines if file is listed in DATASET_COLUMNS
            # If so, headers will be supplied via the dictionary DATASET_COLUMNS
            col_names = DATASET_COLUMNS[filename]
            df = pd.read_csv(filepath, header=None, names=col_names)
        else:
            # For everything else, assume that it has a header row
            df = pd.read_csv(filepath)
        
        # Optionally drop columns if not mentioned in user input
        if drop_columns:
            df = df.drop(columns=drop_columns, errors='ignore', inplace = True)
            
        # Addresses issue with reading breast-cancer-wisconsin data file by separating columns with comma
        if "breast-cancer-wisconsin.data" in filename.lower():
            col_names = [
                "Sample_code_number", 
                "Clump_Thickness", 
                "Uniformity_Cell_Size",
                "Uniformity_Cell_Shape", 
                "Marginal_Adhesion",
                "Single_Epithelial_Cell_Size", 
                "Bare_Nuclei", 
                "Bland_Chromatin",
                "Normal_Nucleoli", 
                "Mitoses", 
                "Class"
            ]
            df = pd.read_csv(
                filepath,
                sep=",",         # comma-delimited
                header=None,     # no header row in the file
                names=col_names, # supply our own column names
                na_values="?"    # treat '?' as a missing value
            )
            # Optionally drop the ID column:
            df.drop(columns=["Sample_code_number"], inplace=True, errors="ignore")
        
        # Addresses issue with reading machine data file
        if "machine.data" in filename:
            # Drop vendor/model/ERP for machine data
            df.drop(columns=["vendor name", "Model Name", "ERP"],
                    inplace=True, errors="ignore")
            
        # Remember the target column
        self.target_column = target_column
        return df
    
    def handle_missing (self, df):
        
        """
        
        Handles missing values. 
        For numerical columns, it will impute with mean. 
        For categorical columns, it will impute with mode.
        
        """
        
        df = df.copy()
        if self.numeric_cols:
            for col in self.numeric_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].mean(), inplace = True)
        if self.categorical_cols:
            for col in self.categorical_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].mode()[0], inplace = True)
        return df
    
    def fit_normalization (self, df):
        
        """
        
        Calculates and stores the mean and std for each numeric column
        For use in normalization

        """
        
        if self.numeric_cols:
            for col in self.numeric_cols:
                self.numeric_means[col] = df[col].mean()
                self.numeric_stds[col] = df[col].std()
    
    def normalize (self, df):
        
        """
        
        Applies z-score normalization to numeric columns
        
        """
        
        df = df.copy()
        if self.numeric_cols:
            for col in self.numeric_cols:
                mean_val = self.numeric_means.get(col,0)
                std_val = self.numeric_stds.get(col, 1e-8)
                df[col] = (df[col] - mean_val) / (std_val + 1e-8)
        return df
    
    def fit_encode (self, df):
        
        """
        
        Determines unique levels for each categorical column

        """
        
        if self.categorical_cols:
            for col in self.categorical_cols:
                self.cat_levels[col] = sorted(df[col].unique())
                
    def encode (self, df):
        
        """
        
        Applies one-hot entcoding to categorical columns

        """
        
        df = df.copy()
        if self.categorical_cols:
            for col in self.categorical_cols:
                levels = self.cat_levels[col]
                for level in levels:
                    new_col = f"{col}_{level}"
                    df[new_col] = (df[col] == level).astype(int)
                df.drop(columns = [col], inplace = True)
        return df
    
    def preprocess (self, df, fit = False):
        
        """
        
        Runs the full preprocessing pipeline by handling missing values, 
        fit normalization, and encoding, if needed.

        """
        
        df = self.handle_missing(df)
        if fit:
            self.fit_normalization(df)
            self.fit_encode(df)
        df = self.normalize (df)
        df = self.encode (df)
        return df
    