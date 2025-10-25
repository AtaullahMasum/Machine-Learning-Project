from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class FillMissingTransformer(BaseEstimator, TransformerMixin):
   """
    Custom transformer to handle common 'None' semantics and numeric imputations.
    """
    
