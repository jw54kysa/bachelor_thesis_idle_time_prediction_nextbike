import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit

idle_time_data = pd.read_csv('../data/df_points/df_points_18_21_class.csv')
from sklearn.preprocessing import StandardScaler

TargetVariable = ['idle_time']
Predictors = ['bike_id', 'lat', 'lng', 'temp', 'rain', 'snow', 'dt_start', 'hex_enc', 'start_min', 'month', 'day']

X = idle_time_data[Predictors].values
y = idle_time_data[TargetVariable].values

## Scale Data
PredictorScaler = StandardScaler()
PredictorScalerFit = PredictorScaler.fit(X)
X = PredictorScalerFit.transform(X)

TargetScaler = StandardScaler()
TargetScalerFit = TargetScaler.fit(y)
y = TargetScalerFit.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)
# , random_state=42

from sklearn import metrics
import numpy as np
from sklearn.metrics import zero_one_loss, accuracy_score
from sklearn.model_selection import cross_validate, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import wandb
import wandb

sweep_configuration_rfr = {
    "project": "RandomForestRegressor",
    "name": "my-awesome-sweep",
    "metric": {"name": "accuracy", "goal": "maximize"},
    "method": "random",
    "parameters": {
        "n_estimators": {
            "values": [int(x) for x in np.linspace(start = 50, stop = 2000, num = 10)]
        },
        "criterion": {
            "values": ['squared_error','absolute_error','poisson']
        },
        "max_depth": {
            "values": [5,10,20,30,40,50,60,70,80,90,100,None]
        },
        "bootstrap": {
            "values": [True, False]
        },
        "max_features": {
            "values": ['auto', 'sqrt', 'log2']
        },
        "min_samples_leaf": {
            "values": [1, 2, 4, 10, 20, 50, 100, 200, 1000]
        },
        "min_samples_split": {
            "values": [2, 5, 10, 100, 200]
        }
    }
}

def my_train_func():
    wandb.init()

    _n_estimators = wandb.config.n_estimators
    _criterion = wandb.config.criterion
    _max_depth = wandb.config.max_depth
    _bootstrap = wandb.config.bootstrap
    _max_features = wandb.config.max_features
    _min_samples_leaf = wandb.config.min_samples_leaf
    _min_samples_split = wandb.config.min_samples_split

    model = RandomForestRegressor(n_estimators=_n_estimators,
                                   criterion=_criterion,
                                   max_depth=_max_depth,
                                   bootstrap=_bootstrap,
                                   max_features=_max_features,
                                   min_samples_leaf=_min_samples_leaf,
                                   min_samples_split=_min_samples_split)

    model.fit(X_train,y_train.ravel())
    y_pred = model.predict(X_test)

    score_training = model.score(X_train, y_train.ravel())
    score_validation = model.score(X_test, y_test.ravel())
    rmse = metrics.mean_squared_error(y_test.ravel(),y_pred.ravel())
    acc = zero_one_loss(y_test.ravel(), y_pred.ravel())
    loss = accuracy_score(y_test.ravel(), y_pred.ravel())

    #wandb.sklearn.plot_feature_importances(model, Predictors)

    wandb.log({"accuracy": acc})
    wandb.log({"conf_matrix": wandb.plot.confusion_matrix(y_true=y_test.ravel(), preds=y_pred.ravel())})
    wandb.log({"feature_imp": wandb.sklearn.plot_feature_importances(model, Predictors)})
    wandb.log({"loss": loss})
    wandb.log({"score_training": score_training, "score_validation":score_validation, "rmse":rmse})

# INIT SWEEP
sweep_id_rfc = wandb.sweep(sweep_configuration_rfr, project="RandomForestRegressor")
# RUN SWEEP
wandb.agent(sweep_id_rfc, function=my_train_func)