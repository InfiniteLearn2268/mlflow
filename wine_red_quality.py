# Wine quality example (red wine) tracked with MLflow on DagsHub.
# Data: P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties.
# Decision Support Systems, Elsevier, 47(4):547-553, 2009.
#
# Run:  python app.py              (alpha=0.5, l1_ratio=0.5)
#       python app.py 0.9 0.1      (alpha=0.9, l1_ratio=0.1)

import logging
import os
import sys
import warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# ---- send runs to DagsHub (set USE_DAGSHUB=0 to log locally instead) ----
if os.getenv("USE_DAGSHUB", "1") == "1":
    import dagshub
    dagshub.init(repo_owner="InfiniteLearn2268", repo_name="mlflow", mlflow=True)

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL (no local file needed)
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )
        sys.exit(1)

    # Split the data into training and test sets (75% / 25%)
    train, test = train_test_split(data, random_state=40)

    # The predicted column is "quality"
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # Settings from the command line, with defaults
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    mlflow.set_experiment("wine-quality")

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print(f"Elasticnet model (alpha={alpha:f}, l1_ratio={l1_ratio:f}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        # parameters = the settings you chose
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        # metrics = the results
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        predictions = lr.predict(train_x)
        signature = infer_signature(train_x, predictions)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # The model registry does not work with a local file store
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                lr,
                name="model",
                registered_model_name="ElasticnetWineModel",
                signature=signature,
                serialization_format="cloudpickle",
            )
        else:
            mlflow.sklearn.log_model(
                lr, name="model", signature=signature, serialization_format="cloudpickle"
            )