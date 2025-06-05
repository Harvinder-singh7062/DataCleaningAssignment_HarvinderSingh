# import all necessary libraries here
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Impute Missing Values
def impute_missing_values(data, strategy='mean'):
    """
    Fill missing values in the dataset.
    :param data: pandas DataFrame
    :param strategy: str, imputation method ('mean', 'median', 'mode')
    :return: pandas DataFrame
    """
    data_copy = data.copy()
    for col in data_copy.columns:
        if data_copy[col].isnull().sum() > 0:
            if data_copy[col].dtype in [np.float64, np.int64]:
                if strategy == 'mean':
                    data_copy[col].fillna(data_copy[col].mean(), inplace=True)
                elif strategy == 'median':
                    data_copy[col].fillna(data_copy[col].median(), inplace=True)
                elif strategy == 'mode':
                    data_copy[col].fillna(data_copy[col].mode()[0], inplace=True)
            else:
                data_copy[col].fillna(data_copy[col].mode()[0], inplace=True)
    return data_copy

# 2. Remove Duplicates
def remove_duplicates(data):
    """
    Remove duplicate rows from the dataset.
    :param data: pandas DataFrame
    :return: pandas DataFrame
    """
    return data.drop_duplicates()

# 3. Normalize Numerical Data
def normalize_data(data, method='minmax'):
    """Apply normalization to numerical features.
    :param data: pandas DataFrame
    :param method: str, normalization method ('minmax' or 'standard')
    :return: pandas DataFrame
    """
    data_copy = data.copy()
    num_cols = data_copy.select_dtypes(include=[np.number]).columns

    if method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    data_copy[num_cols] = scaler.fit_transform(data_copy[num_cols])
    return data_copy

# 4. Remove Redundant Features   
def remove_redundant_features(data, threshold=0.9):
    """Remove redundant or duplicate columns.
    :param data: pandas DataFrame
    :param threshold: float, correlation threshold
    :return: pandas DataFrame
    """
    corr_matrix = data.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    return data.drop(columns=to_drop)
