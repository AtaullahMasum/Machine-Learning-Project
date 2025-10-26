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


