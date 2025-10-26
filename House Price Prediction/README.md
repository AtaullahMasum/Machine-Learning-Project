# 🏠 House Price Prediction
A comprehensive machine learning project that predicts house prices using various regression models and advanced feature engineering techniques. This end-to-end solution includes data preprocessing, model training, hyperparameter tuning, and deployment.
## 📋 Table of Contents
* Project Overview

* Features

* Installation

* dataset

* Project Structure

* Usage

* Models

* Results

* Deployment

* Contributing


## 🎯 Project Overview
This project implements a complete machine learning pipeline for predicting house prices based on various property features. The solution includes:

* Data preprocessing and feature engineering

* Multiple machine learning models with performance comparison

* Hyperparameter tuning using RandomizedSearchCV and GridSearchCV

* Model serialization for production use

* Web interface using Streamlit for easy predictions

## ✨ Features

* 🏗️ Custom preprocessing pipelines with reusable transformers

* 📊 Comprehensive feature engineering (numeric + categorical)

* 🔧 Multiple regression models (Linear, Tree-based, Ensemble)

* ⚡ Hyperparameter optimization for best performance

* 🌐 Streamlit web interface for easy predictions

* 💾 Model persistence with joblib

* 📈 Cross-validation and performance metrics

## 🚀 Installation

## Prerequisites

* Python 3.8+

* pip package manager

## Setup

1. Clone the repository

```bash
git clone <repository-url>
cd "House Price Prediction"
```

2. ### Create virtual environment (Windows)

```bash
python -m venv venv
venv\Scripts\activate
```
3. ### Install dependencies

```bash
pip install -r requirements.txt
```

## equired Packages

```txt
numpy
pandas
scikit-learn
xgboost
joblib
streamlit
```

## 📊 Dataset
### Features
#### Numerical Features:

* area - Property area in square feet

* bedrooms - Number of bedrooms

* bathrooms - Number of bathrooms

* stories - Number of stories

* parking - Number of parking spaces

#### Categorical Features:

* mainroad - Access to main road (yes/no)

* guestroom - Guest room availability (yes/no)

* basement - Basement availability (yes/no)

* hotwaterheating - Hot water heating (yes/no)

* airconditioning - Air conditioning (yes/no)

* prefarea - Preferred area location (yes/no)

* furnishingstatus - Furnishing status (furnished/semi-furnished/unfurnished)

#### Target Variable:

* price - House price in local currency


# 📁 Project Structure

```txt
House Price Prediction/
├── 📁 data/
│   ├── train.csv                  # Training dataset
│   ├── test.csv                   # Test dataset
│   └── submission.csv             # Prediction outputs
├── 📁 models/
│   └── xgb_pipeline.joblib        # Trained model pipeline
├── 📁 notebooks/
│   └── eda_and_model.ipynb        # explorary data Analysis File
├── 📁 src/
|   |__ all_step_in_one_file.ipynb  # Main analysis notebook  
│   ├── data_utils.py               # Data loading & transformation utilities
│   └── features.py                 # Custom transformers & feature engineering
|   |__ train.py                    # Model training 
|   |__ hyperparametertuning.py     # Hyperparameter tuning file 
|── README.md                        # Project documentation
|___ requirement.txt
|___ Docker

```
## 💻 Usage

### Training the Model

```python
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# Load data
train = pd.read_csv('../data/train.csv')

# Prepare features and target
X = train.drop(columns=['price', 'price_log'])
y = train['price_log']

# Train model (example with XGBoost)
pipe_xgb.fit(X, y)

# Save model
joblib.dump(pipe_xgb, "../models/xgb_pipeline.joblib")
```

### Making Predictions

```python
# Load trained model
model = joblib.load("../models/xgb_pipeline.joblib")

# Make predictions
def predict(df):
    preds_log = model.predict(df)
    preds = inv_log_target(preds_log)  # Convert from log space
    return preds

# Example usage
test_data = pd.read_csv('../data/test.csv')
predictions = predict(test_data)
```


## Web Interface

#### Launch the Streamlit app:

```bash
streamlit run src/app_streamlit.py
```

The web interface allows:

* 📤 Uploading CSV files with house features

* 🔮 Getting instant price predictions

* 📥 Downloading results as CSV

## 🤖 Models Implemented

### Linear Models

* LinearRegression - Basic linear regression

* Ridge - L2 regularization

* Lasso - L1 regularization

* ElasticNet - Combined L1 + L2 regularization

* BayesianRidge - Bayesian ridge regression

### Tree-based & Ensemble Models
* XGBRegressor - Gradient boosting (best performer)

* RandomForestRegressor - Random forest

* GradientBoostingRegressor - Gradient boosting

* AdaBoostRegressor - Adaptive boosting

### Pipeline Architecture

```python
# Custom preprocessing pipeline
numeric_pipeline = Pipeline([
    ('num_selector', NumericSelector(numeric_feats)),
    ('fill_missing', FillMissingTransformer()),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('cat_selector', CatagoricalSelector(categorical_feats)),
    ('imputer', FillMissingTransformer()),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combined preprocessor
preprocessor = ColumnTransformer([
    ('numeric', numeric_pipeline, numeric_feats),
    ('categorical', categorical_pipeline, categorical_feats)
])
```

## 📈 Results

### Model Performance (Cross-Validation RMSE)

Model	           Log-space RMSE
Ridge	            0.199
XGBoost	          0.224
Linear Regression	0.201
Lasso	            0.200

### Hyperparameter Tuning

The project includes comprehensive hyperparameter optimization:

```python
📈 Results
Model Performance (Cross-Validation RMSE)
Model	Log-space RMSE
Ridge	0.199
XGBoost	0.224
Linear Regression	0.201
Lasso	0.200
Hyperparameter Tuning
The project includes comprehensive hyperparameter optimization:
```

## 🌐 Deployment

### Streamlit Web App

The project includes a user-friendly web interface:

```python
import streamlit as st
import joblib
from data_utils import inv_log_target

st.title("House Price Predictor")
model = joblib.load("../models/xgb_pipeline.joblib")

# File upload and prediction interface
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    predictions = inv_log_target(model.predict(df))
    st.write("Price Predictions:", predictions)
```

#### Features:

* ✅ Drag-and-drop CSV upload

* ✅ Instant predictions

* ✅ Results display and download


## 🔧 Custom Components

#### Data Utilities (data_utils.py)
* load_data() - Data loading functions

* log_target() - Log transformation for prices

* inv_log_target() - Inverse log transformation

#### Feature Engineering (features.py)

* NumericSelector - Custom numeric feature selector

* CatagoricalSelector - Custom categorical feature selector

* FillMissingTransformer - Advanced missing value imputation

#### 🚀 Future Enhancements

##### Planned 

1. ###### Advanced Feature Engineering

   * Polynomial features

   * Feature interactions

   * Domain-specific transformations

2. ###### Model Enhancements

   * Neural networks

   * Stacking ensembles

   * Automated feature selection

3. ###### Deployment

   * REST API with FastAPI

   * Database integration

   * Real-time prediction service

4. ###### Monitoring

   * Model performance tracking

   * Data drift detection

   * Automated retraining

## 👥 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

* 🐛 Bug fixes

* 💡 New features

* 📚 Documentation improvements

* 🧪 Additional models or techniques

## 📄 License

This project is developed for Advanced  learning purposes as part of a machine learning curriculum.


###### Built with ❤️ using Python, Scikit-learn, XGBoost, and Streamlit

For questions or support, please open an issue in the repository.