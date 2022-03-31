import wandb
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import random

WANDB_PROJECT_NAME = "mlpr_test"

with wandb.init(project=WANDB_PROJECT_NAME):
    # Import Data
    df = pd.read_csv('../../../data/df_points/df_points_18_21.csv')

    # define Target and Predictors
    TargetVariable = ['idle_time']
    Predictors = ['bike_id', 'lat', 'lng', 'temp', 'rain', 'snow', 'dt_start', 'hex_enc', 'start_min', 'month', 'day']
    X = df[Predictors].values
    y = df[TargetVariable].values

    # scale predictor features
    # Sandardization of data
    PredictorScaler = StandardScaler()
    TargetVarScaler = StandardScaler()
    PredictorScalerFit = PredictorScaler.fit(X)
    TargetVarScalerFit = TargetVarScaler.fit(y)
    X = PredictorScalerFit.transform(X)
    y = TargetVarScalerFit.transform(y)

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # import sweep config
    #config = wandb.config

    # define model
    mlpr = MLPRegressor(hidden_layer_sizes=64,
                        activation="relu",
                        solver="sgd",
                        alpha=0.1,
                        learning_rate="adaptive",
                        momentum=1,
                        learning_rate_init=0.8,
                        early_stopping=True)

    # fit and predict
    mlpr.fit(X_train, y_train.ravel())
    y_pred = mlpr.predict(X_test)

    r2 = r2_score(y_test.ravel(), y_pred.ravel())
    mse = mean_squared_error(y_test.ravel(), y_pred.ravel())

    wandb.log({"r2": r2})
    wandb.log({"mse": mse})

    wandb.log({"loss": mlpr.best_loss_})
    wandb.log({'accuracy': mlpr.score(X_test, y_test.ravel())})
