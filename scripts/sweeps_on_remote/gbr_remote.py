import pandas as pd
from numpy import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import wandb

idle_time_data = pd.read_csv('../../data/final_df_points_18_21_class.csv')

TargetVariable = ['idle_time']
Predictors = ['bike_id', 'lat', 'lng', 'temp', 'rain', 'snow', 'wind_speed', 'humidity', 'dt_start'
    , 'hex_enc', 'start_min', 'year', 'month', 'day', 'on_station', 'in_zone', 'zone_name_enc']

X = idle_time_data[Predictors].values
y = idle_time_data[TargetVariable].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, shuffle=False)

sweep_configuration_rfr = {
    "project": "GB-Regression",
    "name": "my-awesome-sweep",
    "metric": {"name": "r2_score", "goal": "maximize"},
    "method": "random",
    "parameters": {
        "loss": {
            "values": ['squared_error', 'absolute_error', 'huber', 'quantile']
        },
        "learning_rate": {
            "values": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.13,0.15,0.2]
        },
        "n_estimators": {
            "values": [8, 14, 19, 24, 32, 41, 54, 67, 75, 89, 102, 123]
        },
        "criterion": {
            "values": ['friedman_mse', 'squared_error', 'mse']
        },
        "alpha": {
            "values": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
        },
        "max_depth": {
            "values": [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, None]
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
    _loss = wandb.config.loss
    _alpha = wandb.config.alpha
    _learning_rate = wandb.config.learning_rate

    _criterion = wandb.config.criterion
    _max_depth = wandb.config.max_depth
    _max_features = wandb.config.max_features
    _min_samples_leaf = wandb.config.min_samples_leaf
    _min_samples_split = wandb.config.min_samples_split

    model = GradientBoostingRegressor(n_estimators=_n_estimators,
                                    loss = _loss,
                                    alpha = _alpha,
                                    learning_rate = _learning_rate,
                                    criterion=_criterion,
                                    max_depth=_max_depth,
                                    max_features=_max_features,
                                    min_samples_leaf=_min_samples_leaf,
                                    min_samples_split=_min_samples_split,)

    model.fit(X_train, y_train.ravel())
    y_pred = model.predict(X_test)

    r2, mse, rmse, mae =eval_regression(y_test, y_pred)

    wandb.log({"r2_score": r2, "MSE": mse, "RMSE": rmse, "MAE" : mae})

# INIT SWEEP
sweep_id_rfc = wandb.sweep(sweep_configuration_rfr, project="GB-Regression")
# RUN SWEEP
wandb.agent(sweep_id_rfc, function=my_train_func)
