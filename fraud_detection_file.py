import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)

# ---------------------------------------------------------------- tracking
# Set USE_DAGSHUB=0 to log to a local ./mlruns folder instead of DagsHub.
if os.getenv('USE_DAGSHUB', '1') == '1':
    import dagshub
    dagshub.init(repo_owner='InfiniteLearn2268', repo_name='mlflow', mlflow=True)
mlflow.set_experiment('fraud-detection')

# ---------------------------------------------------------------- data
DATA_PATH = 'fraud_detection.csv'      # keep this file next to train.py
df = pd.read_csv(DATA_PATH)

y = df['isFraud']
X = df.drop(columns=['isFraud', 'TransactionID'])   # ID is only a label, not a feature

cat_cols = [c for c in X.columns if X[c].dtype == 'object' or str(X[c].dtype) == 'str']
num_cols = [c for c in X.columns if c not in cat_cols]

# stratify keeps the same fraud share (about 3%) in both train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ---------------------------------------------------------------- preprocessing
def make_preprocessor(scale):
    num_steps = [('impute', SimpleImputer(strategy='median'))]
    if scale:
        num_steps.append(('scale', StandardScaler()))
    return ColumnTransformer([
        ('num', Pipeline(num_steps), num_cols),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols),
    ])

# ---------------------------------------------------------------- experiments
# Each entry is one run. Add or change entries to try more ideas.
experiments = [
    ('logreg_C1',  'LogisticRegression', True,  {'C': 1.0,  'class_weight': 'balanced', 'max_iter': 1000}),
    ('logreg_C01', 'LogisticRegression', True,  {'C': 0.1,  'class_weight': 'balanced', 'max_iter': 1000}),
    ('rf_100_d5',  'RandomForest',       False, {'n_estimators': 100, 'max_depth': 5,  'class_weight': 'balanced', 'random_state': 42}),
    ('rf_300_d10', 'RandomForest',       False, {'n_estimators': 300, 'max_depth': 10, 'class_weight': 'balanced', 'random_state': 42}),
]
models = {'LogisticRegression': LogisticRegression, 'RandomForest': RandomForestClassifier}

for run_name, model_name, scale, params in experiments:
    with mlflow.start_run(run_name=run_name):
        pipe = Pipeline([('prep', make_preprocessor(scale)),
                         ('model', models[model_name](**params))])
        pipe.fit(X_train, y_train)

        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

        # parameters = the settings you chose
        mlflow.log_param('model', model_name)
        mlflow.log_param('test_size', 0.2)
        mlflow.log_params(params)

        # metrics = the results (accuracy is skipped on purpose: fraud is only ~3% of rows)
        mlflow.log_metric('recall',       recall_score(y_test, pred, zero_division=0))
        mlflow.log_metric('precision',    precision_score(y_test, pred, zero_division=0))
        mlflow.log_metric('f1',           f1_score(y_test, pred, zero_division=0))
        mlflow.log_metric('roc_auc',      roc_auc_score(y_test, proba))
        mlflow.log_metric('pr_auc',       average_precision_score(y_test, proba))
        mlflow.log_metric('true_positives',  int(tp))
        mlflow.log_metric('false_positives', int(fp))
        mlflow.log_metric('false_negatives', int(fn))

        # artifact = the trained model (preprocessing included)
        mlflow.sklearn.log_model(pipe, 'model', serialization_format='cloudpickle')

        print(f'{run_name:12s} recall={recall_score(y_test, pred):.2f} '
              f'precision={precision_score(y_test, pred, zero_division=0):.2f} '
              f'roc_auc={roc_auc_score(y_test, proba):.3f}')