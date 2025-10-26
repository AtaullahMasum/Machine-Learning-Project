import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, kFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, StackingRegressor, VotingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, LassoLarsIC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
import joblib
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error

from data_utils import load_data, log_target, inv_log_target
from features import FillMissingTransformer, NumericSelector, CatagoricalSelector

# load data
train  =  pd.read_csv('../data/train.csv')
test  =  pd.read_csv('../data/test.csv')

# target log transformation
train["price_log"] = log_target(train["price"])

# 4. Feature Engineering Pipelines
numeric_feats = train.select_dtypes(include=[np.number]).columns.tolist()
categorical_feats = train.select_dtypes(include=['object']).columns.tolist()
numeric_feats.remove('price_log')
numeric_feats.remove('price')


numeric_pipeline = Pipeline(steps=[
  ('num_selector', NumericSelector(numeric_feats)),
  ('fill_missing', FillMissingTransformer()),
  ('scaler', StandardScaler())
])
categorical_pipeline = Pipeline(steps=[
  ('cat_selector', CatagoricalSelector(categorical_feats)),
  ('imputer', FillMissingTransformer()),
  ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
###  5. ColumnTransformer to Combine
preprocessor = ColumnTransformer(
    transformers=[
      ('numeric', numeric_pipeline, numeric_feats),
      ('categorical', categorical_pipeline, categorical_feats)
    ], remainder='drop')


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

# 7. Model Definitions and Pipelines
ridge = Ridge(alpha=1.0, random_state=42)
linear = LinearRegression()
lasso = Lasso(alpha=0.01)
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
bayesianridge  = BayesianRidge()
xgb = XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',
    random_state=42,
    n_jobs=1
)


pipe_ridge = Pipeline(steps=[
    ('preprocessor', preprocessor1),('ridge', ridge)
    ])
pipe_linear = Pipeline(steps=[
  ('preprocessor', preprocessor1),('linear', linear)
])
pipe_lasso = Pipeline(steps = [
  ('preprocessor', preprocessor1),('lasso', lasso)
])
pipe_elastic_net = Pipeline(steps = [
  ('preprocessor', preprocessor1),('elastic_net', elastic_net)
])
pipe_bayesianridge = Pipeline(steps = [
  ('preprocessor', preprocessor1),('bayesianridge', bayesianridge)
])
pipe_xgb = Pipeline(steps=[
  ('preprocessor', preprocessor1),('xgb', xgb)
])


X = train.drop(columns=['price','price_log'])
y = train['price_log']


#8. Model Evaluation with Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)


def rmse_cv(pipe):
  scores = - cross_val_score(pipe, X, y, scoring="neg_root_mean_squared_error", cv = kf, n_jobs= -1)
  return scores

print("CV Ridge RMSE (log-space):", rmse_cv(pipe_ridge).mean())
print("CV XGB RMSE (log-space):", rmse_cv(pipe_xgb).mean())

pipe_ridge.fit(X,y)
pipe_linear.fit(X,y)
pipe_lasso.fit(X,y)
pipe_elastic_net.fit(X,y)
pipe_bayesianridge.fit(X,y)
pipe_xgb.fit(X, y)


#10. model saving
joblib.dump(pipe_xgb,"../models/xgb_pipeline.joblib" )
print("Saved model to ../models/xgb_pipeline.joblib")


model = joblib.load("../models/xgb_pipeline.joblib")
# Prediction function
def predict(df):
  preds_log = model.predict(df)
  preds = inv_log_target(preds_log)
  return preds


if __name__ == "__main__":
  test = pd.read_csv('../data/test.csv')
  test_x, test_y = test.drop(columns=['price']), test['price']
  preds = predict(test_x)
  test_ids = range(len(test_x))
  out = pd.DataFrame({'Id': test_ids, 'PredictedPrice': preds[:]})
  out.to_csv('../data/submission.csv', index=False)
  for i in range(10):
    print(f"Predicted price: {preds[i]:.2f}, Actual price: {test_y.iloc[i]:.2f}")
