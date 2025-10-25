from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class FillMissingTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to handle common 'None' semantics and numeric imputations.
    """
    def __init__(self, numeric_fill = None, cat_fill='None'):
        self.numeric_fill = numeric_fill
        self.cat_fill = cat_fill
    def fit(self, X, y=None):
        if self.numeric_fill is None:
            self.numeric_stats_ = X.select_dtypes(include=[np.number]).median()
        else:
            self.numeric_stats_ = self.numeric_fill
        return self
    def transform(self, X):
        X = X.copy()
        # Fill categorical missing with 'None'
        for c in X.select_dtypes(include=['object']).columns:
            X[c] = X[c].fillna(self.cat_fill)
        # Fill numeric missing with median or specified value
        for c, v in self.numeric_stats_.items():
            if c in X.columns:
                X[c] = X[c].fillna(v)
        return X
class NumericSelector(BaseEstimator, TransformerMixin):
    """
    Custom transformer to select numeric columns from a DataFrame.
    """
    def __init__(self, numeric_cols):
        self.numeric_cols = numeric_cols
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.numeric_cols]
class CatagoricalSelector(BaseEstimator, TransformerMixin):
    """
    Custom transformer to select categorical columns from a DataFrame.
    """
    def __init__(self, categorical_cols):
        self.categorical_cols = categorical_cols
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.categorical_cols]
          
        