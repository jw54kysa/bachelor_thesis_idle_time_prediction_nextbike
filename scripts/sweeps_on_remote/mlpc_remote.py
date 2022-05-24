import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import zero_one_loss, accuracy_score
from sklearn.neural_network import MLPClassifier
import wandb
import random

import os
os.environ["WANDB_CONSOLE"] = "off"

idle_time_data = pd.read_csv('../../data/final_df_points_18_21_class.csv')

TargetVariable = ['idle_time']
Predictors = ['bike_id', 'lat', 'lng', 'temp', 'rain', 'snow', 'wind_speed', 'humidity', 'dt_start',
              'hex_enc', 'start_min', 'year', 'month', 'day', 'on_station', 'in_zone', 'zone_name_enc']

X = idle_time_data[Predictors].values
y = idle_time_data[TargetVariable].values

PredictorScaler = StandardScaler()
PredictorScalerFit = PredictorScaler.fit(X)
X = PredictorScalerFit.transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, shuffle=False)

sweep_configuration = {
    "project": "MultiLayerPerceptronClassification",
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

    model = MLPClassifier(hidden_layer_sizes=_hidden_layer_sizes,
                         activation=_activation,
                         solver=_solver,
                         alpha=_alpha,
                         learning_rate=_learning_rate,
                         momentum=_momentum,
                         early_stopping=True)

    model.fit(X_train, y_train.ravel())
    y_pred = model.predict(X_test)

    score_training = model.score(X_train, y_train.ravel())
    score_validation = model.score(X_test, y_test.ravel())

    acc = accuracy_score(y_test.ravel(), y_pred.ravel())
    loss = zero_one_loss(y_test.ravel(), y_pred.ravel())


    wandb.log({"accuracy": acc, "loss": loss})
    wandb.log({"conf_matrix": wandb.plot.confusion_matrix(y_true=y_test.ravel(), preds=y_pred.ravel())})
    wandb.log({"score_training": score_training, "score_validation": score_validation})


# INIT SWEEP
sweep_id_rfc = wandb.sweep(sweep_configuration, project="MultiLayerPerceptronClassification")
# RUN SWEEP
wandb.agent(sweep_id_rfc, function=my_train_func)
