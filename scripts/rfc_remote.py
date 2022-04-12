import pandas as pd
from sklearn.metrics import zero_one_loss, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import wandb

idle_time_data = pd.read_csv('../data/final_df_points_18_21_class.csv')

TargetVariable = ['idle_time_class']
Predictors = ['bike_id', 'lat', 'lng', 'temp', 'rain', 'snow', 'wind_speed', 'humidity', 'dt_start',
              'hex_enc', 'start_min', 'year', 'month', 'day', 'on_station', 'in_zone', 'zone_name_enc']

X = idle_time_data[Predictors].values
y = idle_time_data[TargetVariable].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, shuffle=False)

sweep_configuration = {
    "project": "RandomForestClass",
    "name": "RFC-sweep-split-inorder-not-scaled",
    "metric": {"name": "accuracy", "goal": "maximize"},
    "method": "random",
    "parameters": {
        "n_estimators": {
            "values": [8, 16, 32, 64, 128, 256, 512, 640, 768]
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

    score_training = model.score(X_train, y_train.ravel())
    score_validation = model.score(X_test, y_test.ravel())

    acc = accuracy_score(y_test.ravel(), y_pred.ravel())
    loss = zero_one_loss(y_test.ravel(), y_pred.ravel())

    wandb.sklearn.plot_feature_importances(model, Predictors)

    wandb.log({"accuracy": acc})
    wandb.log({"conf_matrix": wandb.plot.confusion_matrix(y_true=y_test.ravel(), preds=y_pred.ravel())})
    wandb.log({"feature_imp": wandb.sklearn.plot_feature_importances(model, Predictors)})
    wandb.log({"loss": loss})
    wandb.log({"score_training": score_training, "score_validation": score_validation})


# INIT SWEEP
sweep_id_rfc = wandb.sweep(sweep_configuration, project="RandomForestClassifier")
# RUN SWEEP
wandb.agent(sweep_id_rfc, function=my_train_func)
