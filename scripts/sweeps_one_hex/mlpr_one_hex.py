from math import sqrt

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import wandb
import random

idle_time_data = pd.read_csv('../../data/final_df_points_18_21_class.csv')

#Augustusplatz
idle_time_data = idle_time_data[idle_time_data['hex_id'] == '881f1a8cb7fffff']

TargetVariable = ['idle_time']
Predictors = ['bike_id', 'lat', 'lng', 'temp', 'rain', 'snow', 'wind_speed', 'humidity', 'dt_start',
              'hex_enc', 'start_min', 'year', 'month', 'day', 'on_station', 'in_zone', 'zone_name_enc']

X = idle_time_data[Predictors].values
y = idle_time_data[TargetVariable].values

PredictorScaler = StandardScaler()
PredictorScalerFit = PredictorScaler.fit(X)
X = PredictorScalerFit.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, shuffle=False)

sweep_configuration = {
    "project": "MLP-Regression-One-Hex",
    "name": "MLPC-sweep-new-data",
    "metric": {"name": "r2_score", "goal": "maximize"},
    "method": "random",
    "parameters": {
        "activation": {
            "values": ['identity', 'logistic', 'tanh', 'relu']
        },
        "solver": {
            "values": ['lbfgs', 'sgd', 'adam']
        },
        "alpha": {
            "values": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
        },
        "learning_rate": {
            "values": ['constant', 'invscaling', 'adaptive']
        },
        "momentum": {
            "values": [0.9, 0.8, 0.7, 0.6, 0.5]
        }
    }
}

def eval_regression(y_test,y_pred):
    # Metrics
    # r2, mae, mse, rmse
    r2 = r2_score(y_test, y_pred.ravel())
    mae = mean_absolute_error(y_test, y_pred.ravel())
    mse = mean_squared_error(y_test, y_pred.ravel())
    rmse = sqrt(mse)

    print('r2: %f' % r2)
    print('mae: %f' % mae)
    print('mse: %f' % mse)
    print('rmse: %f' % rmse)

    return r2, mse, rmse


def my_train_func():
    wandb.init()

    hls = [(16), (32), (64), (128),
           (16, 16), (32, 32), (64, 64), (128, 128),
           (16, 32, 16), (32, 64, 32), (64, 128, 64),
           (128, 64, 128)]

    _hidden_layer_sizes = random.choice(hls)
    print(_hidden_layer_sizes)
    _activation = wandb.config.activation
    _solver = wandb.config.solver
    _alpha = wandb.config.alpha
    _learning_rate = wandb.config.learning_rate
    _momentum = wandb.config.momentum

    wandb.config.hidden_layer_sizes = _hidden_layer_sizes

    model = MLPRegressor(hidden_layer_sizes=_hidden_layer_sizes,
                         activation=_activation,
                         solver=_solver,
                         alpha=_alpha,
                         learning_rate=_learning_rate,
                         momentum=_momentum,
                         early_stopping=True,
                         max_iter=600)

    model.fit(X_train, y_train.ravel())
    y_pred = model.predict(X_test)

    r2, mse, rmse = eval_regression(y_test, y_pred)

    wandb.log({"r2_score": r2, "MSE": mse, "RMSE": rmse})


# INIT SWEEP
sweep_id_rfc = wandb.sweep(sweep_configuration, project="MLP-Regression-One-Hex")
# RUN SWEEP
wandb.agent(sweep_id_rfc, function=my_train_func)
