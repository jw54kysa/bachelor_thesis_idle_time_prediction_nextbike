import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import wandb
import random

idle_time_data = pd.read_csv('../../data/final_df_points_18_21_class.csv')

TargetVariable = ['idle_time']
Predictors = ['bike_id', 'lat', 'lng', 'temp', 'rain', 'snow', 'wind_speed', 'humidity', 'dt_start',
              'hex_enc', 'start_min', 'year', 'month', 'day', 'on_station', 'in_zone', 'zone_name_enc']

X = idle_time_data[Predictors].values
y = idle_time_data[TargetVariable].values

PredictorScaler = StandardScaler()
PredictorScalerFit = PredictorScaler.fit(X)
X = PredictorScalerFit.transform(X)

TargetScaler = StandardScaler()
TargetScalerFit = TargetScaler.fit(y)
y = TargetScalerFit.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, shuffle=False)

sweep_configuration = {
    "project": "MultiLayerPerceptronRegression-new-data",
    "name": "MLPC-sweep-new-data",
    "metric": {"name": "accuracy", "goal": "maximize"},
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


def my_train_func():
    wandb.init()

    hls = [(16), (32), (64), (128),
           (16, 16), (32, 32), (64, 64), (128, 128),
           (16, 32, 16), (32, 64, 32), (64, 128, 64),
           (128, 64, 128)]

    _hidden_layer_sizes = random.choice(hls)
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
                         early_stopping=True)

    model.fit(X_train, y_train.ravel())
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred.ravel())
    mse = mean_squared_error(y_test, y_pred.ravel())

    wandb.log({"r2": r2, "mse": mse})


# INIT SWEEP
sweep_id_rfc = wandb.sweep(sweep_configuration, project="MultiLayerPerceptronRegression-new-data")
# RUN SWEEP
wandb.agent(sweep_id_rfc, function=my_train_func)
