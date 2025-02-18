# Nonparametric Learning (Project 1)

Hello! Welcome to my submission for project 1. In this repository, you will find my implementation of a nonparametric k-nearest neighbour algorithm for both classification and regression tasks. It includes a full machine learning pipieline with data preprocessing, cross validation, distance and kernel calculations, baseline null models, and instance selection using condensed nearest neighbour. The goal is to demonstrate that a tuned k-nearest neighbour model significantly outperforms a null model and that data reduction via condensed k-nearest neighbour maintains performance while improving computational efficacy. 

## Project Structure

### datapreprocessor.py
Contains the DataPreprocessor class, which handles loading CSV data (supplying column names for known datasets), imputation of missing values, z-score normalization, and one-hot encoding of categorical features.

### distance_voting.py
Implements the Minkowski distance function and a plurality vote function, which are used in the k-NN classifier.

### k_nearest.py
Contains functions for k-NN classification (knn_classify) and k-NN regression (knn_regress), including the use of a Gaussian (RBF) kernel for regression.

### null.py
Implements simple baseline models: null_classifier (returns the most common class) and null_regressor (returns the mean target value), along with helper functions to run these models.

### condensednn.py
Implements the condensed nearest neighbor algorithm, which iteratively adds misclassified training examples to reduce the training set size without significantly compromising accuracy.

### crossvalidation.py
Provides functions to perform 5×2 cross-validation for both classification (stratified_5x2_cv) and regression (simple_5x2_cv).

### user.py
The main script that ties together the preprocessing, modeling, and evaluation. It prompts the user for inputs (file path, task type, target column, numeric and categorical columns) and then runs cross-validation along with demonstration outputs for null models, k-NN, and condensed k-NN.

## Installation
1. Clone or download the project repository.
2. Install the required packages using pip (if you don't already have them installed): 
```bash
pip install numpy pandas matplotlib 
```
## Usage
### Running the Main Script
Execute the main script by running:

```python
python3 user.py
```
Follow the prompts:

File Path: Enter the path to your CSV dataset.

Task Type: Type classification or regression.

Target Column: Specify the name of the target column.

Numeric Columns: Provide a comma-separated list of numeric feature names.

Categorical Columns: Provide a comma-separated list of categorical feature names.

### Output
The script will perform 5×2 cross-validation and print out performance metrics:

For classification: Fold accuracies, average accuracy, and null model accuracy.

For regression: Fold MSE values, average MSE, and null model MSE. 

Additionally, demonstration outputs will show sample predictions (including the nearest neighbors for a chosen test point) for the null model, k-NN, and condensed k-NN.
