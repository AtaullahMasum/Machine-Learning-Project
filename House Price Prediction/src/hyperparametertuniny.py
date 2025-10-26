# Required imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, make_scorer

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
# target log transformation
train["price_log"] = np.log1p(train["price"])

numeric_feats = train.select_dtypes(include=[np.number]).columns.tolist()
categorical_feats = train.select_dtypes(include=['object']).columns.tolist()
numeric_feats.remove('price_log')
numeric_feats.remove('price')

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent", fill_value="None")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor1 = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_feats),
    ("cat", categorical_transformer, categorical_feats)
], remainder="drop")

X = train.drop(columns=['price', 'price_log'])
y = train['price_log']

# Basic pipeline: scaler + model (we'll swap model in param grid)
pipe = Pipeline([
    ('preprocessor',preprocessor1),    # helps Ridge; harmless for trees
    ('model', Ridge())               # placeholder; will be replaced in GridSearch param_grid
])

# Parameter grid as a LIST of dicts: one dict per estimator type
param_grid = [
    # ---- Ridge options ----
    {
        'model': [Ridge()],
        'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
        'model__solver': ['auto']   # optional
    },

    # ---- XGBoost options ----
    {
        # supply an XGBRegressor instance (any hard-coded defaults you want)
        'model': [XGBRegressor(tree_method='hist', random_state=42, n_jobs=-1, verbosity=0)],
        'model__n_estimators': [100, 300, 600],
        'model__max_depth': [3, 4, 6],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__subsample': [0.6, 0.8, 1.0],
        'model__colsample_bytree': [0.6, 0.8, 1.0]
    }
]

# Scorer: you can use built-in 'neg_root_mean_squared_error' or create one
# Note: GridSearch expects higher-is-better scoring, so RMSE must be negated or use scorer that returns negative
scoring = 'neg_root_mean_squared_error'  # available in recent sklearn versions

# Cross-validation split object
cv_inner = KFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV: does exhaustive search over the param grid
grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=cv_inner,
    scoring=scoring,
    n_jobs=-1,
    verbose=2,
    refit=True  # refit best estimator on the whole training set
)

# Fit on training set only
grid.fit(X, y)

# Best results
test_x, y_true = test.drop(columns=['price']), test['price']
print("Best params:", grid.best_params_)
print("Best CV score (neg RMSE):", grid.best_score_)

# Evaluate the best model on held-out test set
best_model = grid.best_estimator_
y_pred = best_model.predict(test_x)
y_pred = np.expm1(y_pred)  # inverse log transform
rmse_test = mean_squared_error(y_true, y_pred)
print("Test RMSE:", rmse_test)
