import numpy as np
import pandas as pd
import random
from datapreprocessor import DataPreprocessor
from k_nearest import knn_classify, knn_regress
from distance_voting import minkowski_distance, plurality_vote
from condensednn import condensed_nearest_neighbor
from crossvalidation import stratified_5x2_cv, simple_5x2_cv
from null import null_classifier, null_regressor, run_null_model_regression, run_null_model_classification

if __name__ == "__main__":
    # Prompt the user to enter the file path for the CSV data
    file_path = input("Enter the CSV file path: ")

    # Let the user specify whether this is a classification or regression task
    task_type = input("Is this a 'classification' or 'regression' task? ").strip().lower()

    if task_type == 'classification':
        target_col = input("What is the target column (classification)? ").strip()

        # Ask user for names of numeric and categorical columns
        numeric_cols = input("Enter the names of numeric columns (comma-separated): ").strip().split(',')
        numeric_cols = [col.strip() for col in numeric_cols if col.strip()]
        
        categorical_cols = input("Enter the names of categorical columns (comma-separated): ").strip().split(',')
        categorical_cols = [col.strip() for col in categorical_cols if col.strip()]

        # Initialize the preprocessor and load data
        preprocessor = DataPreprocessor(numeric_cols=numeric_cols, categorical_cols=categorical_cols)
        df = preprocessor.load_data(file_path, target_col)
        df = preprocessor.preprocess(df, fit=True)

        # Separate features and target
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values

        # Run 5×2 cross-validation for classification
        folds = stratified_5x2_cv(df, target_col)
        cv_results = []
        for i, (train_df, test_df) in enumerate(folds):
            X_train = train_df.drop(columns=[target_col]).values
            y_train = train_df[target_col].values
            X_test = test_df.drop(columns=[target_col]).values
            y_test = test_df[target_col].values

            # Predict on the test fold
            predictions = []
            for x in X_test:
                pred = knn_classify(X_train, y_train, x, k=3, p=2)
                predictions.append(pred)

            accuracy = np.mean(np.array(predictions) == y_test)
            cv_results.append(accuracy)
            print(f"Fold {i+1} classification accuracy: {accuracy:.3f}")
            
        null_acc = run_null_model_classification(X_train, y_train, X_test, y_test)

        print("Average classification accuracy (5×2 CV):", np.mean(cv_results))
        print(f"Null model accuracy: {null_acc:.3f}")
        
        print("\n--- Classification Predictions ---")
        # Use the last fold's training and test sets for demo purposes.
        demo_X_train = X_train
        demo_y_train = y_train
        demo_X_test = X_test
        demo_y_test = y_test

        # Pick a sample test point (first test instance)
        demo_x = demo_X_test[0]
        demo_true = demo_y_test[0]
        print(f"\nSample test point features: {demo_x}")
        print(f"True label: {demo_true}")

        # Null classifier demonstration (returns most common label from training set)
        demo_null = null_classifier(demo_y_train)
        print(f"Null classifier prediction (most common class): {demo_null}")

        # k-NN classifier demonstration: show neighbors and prediction.
        distances = []
        for i in range(len(demo_X_train)):
            d = minkowski_distance(demo_X_train[i], demo_x, p=2)
            distances.append((d, demo_y_train[i]))
        distances.sort(key=lambda tup: tup[0])
        k = 3
        neighbors = distances[:k]
        print(f"k-NN neighbors (distance, label): {neighbors}")
        demo_knn_pred = knn_classify(demo_X_train, demo_y_train, demo_x, k=k, p=2)
        print(f"k-NN classifier prediction: {demo_knn_pred}")

        # Condensed nearest neighbor demonstration:
        X_condensed, y_condensed = condensed_nearest_neighbor(demo_X_train, demo_y_train, p=2)
        print(f"Condensed training set size: {len(X_condensed)} (from original size {len(demo_X_train)})")
        demo_knn_condensed = knn_classify(X_condensed, y_condensed, demo_x, k=k, p=2)
        print(f"k-NN prediction using condensed set: {demo_knn_condensed}")

    elif task_type == 'regression':
        target_col = input("What is the target column for regression? ").strip()

        # Prompt for numeric and categorical columns
        numeric_cols = input("Enter the names of numeric columns (comma-separated): ").strip().split(',')
        numeric_cols = [col.strip() for col in numeric_cols if col.strip()]

        cat_cols = input("Enter the names of categorical columns (comma-separated): ").strip().split(',')
        cat_cols = [col.strip() for col in cat_cols if col.strip()]

        # Initialize the preprocessor and load data
        preprocessor = DataPreprocessor(numeric_cols=numeric_cols, categorical_cols=cat_cols)
        df = preprocessor.load_data(file_path, target_col)
        df = preprocessor.preprocess(df, fit=True)

        # Separate features and target
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values

        # Run 5×2 cross-validation for regression
        folds = simple_5x2_cv(df, target_col)
        reg_cv_results = []
        for i, (train_df, test_df) in enumerate(folds):
            X_train = train_df.drop(columns=[target_col]).values
            y_train = train_df[target_col].values
            X_test = test_df.drop(columns=[target_col]).values
            y_test = test_df[target_col].values

            predictions = []
            for x in X_test:
                pred = knn_regress(X_train, y_train, x, k=3, p=2, gamma=0.5)
                predictions.append(pred)

            mse = np.mean((np.array(predictions) - y_test)**2)
            reg_cv_results.append(mse)
            print(f"Fold {i+1} regression MSE: {mse:.3f}")
        
        null_mse = run_null_model_regression(X_train, y_train, X_test, y_test)

        print("Average regression MSE (5×2 CV):", np.mean(reg_cv_results))
        print(f"Null model MSE: {null_mse:.3f}")
        
        print("\n--- Regression Predictions ---")
        # Use the last fold's training and test sets for demo purposes.
        demo_X_train = X_train
        demo_y_train = y_train
        demo_X_test = X_test
        demo_y_test = y_test

        # Pick a sample test point (first test instance)
        demo_x = demo_X_test[0]
        demo_true = demo_y_test[0]
        print(f"\nSample test point features: {demo_x}")
        print(f"True target value: {demo_true}")

        # Null regressor demonstration (returns mean of target values from training set)
        demo_null_reg = null_regressor(demo_y_train)
        print(f"Null regressor prediction (mean value): {demo_null_reg:.2f}")

        # k-NN regressor demonstration: show neighbors and prediction.
        distances = []
        for i in range(len(demo_X_train)):
            d = minkowski_distance(demo_X_train[i], demo_x, p=2)
            distances.append((d, demo_y_train[i]))
        distances.sort(key=lambda tup: tup[0])
        k = 3
        neighbors = distances[:k]
        print(f"k-NN neighbors (distance, target value): {neighbors}")
        demo_knn_reg = knn_regress(demo_X_train, demo_y_train, demo_x, k=k, p=2, gamma=0.5)
        print(f"k-NN regressor prediction: {demo_knn_reg:.2f}")

    else:
        print("Invalid task type specified. Please choose either 'classification' or 'regression'.")
