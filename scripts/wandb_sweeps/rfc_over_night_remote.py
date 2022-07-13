import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import wandb

idle_time_data = pd.read_csv('../../data/final_df_points_18_21_class.csv')

TargetVariable = ['over_night']
Predictors = ['bike_id', 'lat', 'lng', 'temp', 'rain', 'snow', 'wind_speed', 'humidity', 'dt_start',
              'hex_enc', 'start_min', 'year', 'month', 'day', 'on_station', 'in_zone', 'zone_name_enc']

X = idle_time_data[Predictors].values
y = idle_time_data[TargetVariable].values

PredictorScaler = StandardScaler()
PredictorScalerFit = PredictorScaler.fit(X)
X = PredictorScalerFit.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, shuffle=False)

from sklearn.metrics import f1_score

sweep_configuration = {
    "project": "Over-Night-RF-Class",
    "name": "my-awesome-sweep",
    "metric": {"name": "macro_recall", "goal": "maximize"},
    "method": "random",
    "parameters": {
        "n_estimators": {
            "values": [8, 14, 19, 24, 32, 41, 54, 65, 78,86,98, 107, 114, 128]
        },
        "criterion": {
            "values": ['entropy', 'gini']
        },
        "max_depth": {
            "values": [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, None]
        },
        "bootstrap": {
            "values": [True, False]
        },
        "max_features": {
            "values": ['auto', 'sqrt', 'log2']
        },
        "min_samples_leaf": {
            "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50, 60]
        },
        "min_samples_split": {
            "values": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 18, 20, 25, 30, 40, 50, 60]
        }
    }
}

def eval_classification(y_test,y_pred,labels):
    # Metrics
    acc = accuracy_score(y_test, y_pred.ravel())
    macro_precision = precision_score(y_test.ravel(), y_pred.ravel(), average='binary', labels=labels)
    macro_recall = recall_score(y_test.ravel(), y_pred.ravel(), average='binary', labels=labels)
    macro_f1 = f1_score(y_test.ravel(), y_pred.ravel(), average='binary', labels=labels)

    print(acc)
    #print(macro_precision)
    #print(macro_recall)
    #print(macro_f1)

    return acc, macro_precision, macro_recall, macro_f1

def my_train_func():
    wandb.init()

    _n_estimators = wandb.config.n_estimators
    _criterion = wandb.config.criterion
    _max_depth = wandb.config.max_depth
    _bootstrap = wandb.config.bootstrap
    _max_features = wandb.config.max_features
    _min_samples_leaf = wandb.config.min_samples_leaf
    _min_samples_split = wandb.config.min_samples_split

    model = RandomForestClassifier(n_estimators=_n_estimators,
                                   criterion=_criterion,
                                   max_depth=_max_depth,
                                   bootstrap=_bootstrap,
                                   max_features=_max_features,
                                   min_samples_leaf=_min_samples_leaf,
                                   min_samples_split=_min_samples_split,
                                   n_jobs=-1)

    model.fit(X_train, y_train.ravel())
    y_pred = model.predict(X_test)

    acc, macro_precision, macro_recall, macro_f1 = eval_classification(y_test,y_pred,[1,2])

    wandb.log({"accuracy": acc})
    wandb.log({"conf_matrix": wandb.plot.confusion_matrix(y_true=y_test.ravel(), preds=y_pred.ravel())})
    wandb.log({"macro_precision": macro_precision})
    wandb.log({"macro_recall": macro_recall})
    wandb.log({"macro_f1": macro_f1})

# WandB silent
import os
os.environ["WANDB_SILENT"] = "true"

# INIT SWEEP
sweep_id_rfc = wandb.sweep(sweep_configuration, project="Over-Night-RF-Class")
# RUN SWEEP
wandb.agent(sweep_id_rfc, function=my_train_func)