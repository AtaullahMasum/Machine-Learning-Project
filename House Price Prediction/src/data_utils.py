import numpy as np
import pandas as pd
 
def load_data(path):
    """Load dataset from a CSV file."""
    return pd.read_csv(path)

def log_target(y):
    """Apply log transformation to the target variable."""
    return np.log1p(y)

def inv_log_target(y):
    """Inverse log transformation."""
    return np.expm1(y)
  
  