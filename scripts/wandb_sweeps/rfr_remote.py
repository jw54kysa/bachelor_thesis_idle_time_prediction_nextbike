import pandas as pd
from numpy import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import wandb

idle_time_data = pd.read_csv('../../data/final_df_points_18_21_class.csv')

TargetVariable = ['idle_time']
Predictors = ['bike_id', 'lat', 'lng', 'temp', 'rain', 'snow', 'wind_speed', 'humidity', 'dt_start'
    , 'hex_enc', 'start_min', 'year', 'month', 'day', 'on_station', 'in_zone', 'zone_name_enc']

X = idle_time_data[Predictors].values
y = idle_time_data[TargetVariable].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, shuffle=False)

sweep_configuration_rfr = {
    "project": "RF-Regression",
    "name": "my-awesome-sweep",
    "metric": {"name": "r2_score", "goal": "maximize"},
    "method": "random",
    "parameters": {
        "n_estimators": {
            "values": [8, 14, 19, 24, 32, 41, 54]
        },
        "criterion": {
            "values": ['squared_error', 'absolute_error', 'poisson']
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
            "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 18, 20, 25, 30]
        },
        "min_samples_split": {
            "values": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 18, 20, 25, 30]
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

    return r2, mse, rmse, mae


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
                                  min_samples_split=_min_samples_split,
                                  n_jobs=2)

    model.fit(X_train, y_train.ravel())
    y_pred = model.predict(X_test)

    r2, mse, rmse, mae =eval_regression(y_test, y_pred)

    wandb.log({"r2_score": r2, "MSE": mse, "RMSE": rmse, "MAE" : mae})

# INIT SWEEP
sweep_id_rfc = wandb.sweep(sweep_configuration_rfr, project="RF-Regression")
# RUN SWEEP
wandb.agent(sweep_id_rfc, function=my_train_func)
