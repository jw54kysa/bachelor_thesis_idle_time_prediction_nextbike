import wandb
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss
from sklearn.preprocessing import StandardScaler

WANDB_PROJECT_NAME = "mlpr_hyperparam_opt"

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
    # without , random_state=42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # import sweep config
    config = wandb.config
    #wandb.config.hidden_layer_sizes = [(10,10), (30,30), (60,60, (90,90), (10,10,10), (30,30,30), (60,60,60), (90,90,90)]
    # define model
    mlpr = MLPRegressor(hidden_layer_sizes=config.hidden_layer_sizes,
                        activation=config.activation,
                        solver=config.solver,
                        alpha=config.alpha,
                        learning_rate=config.learning_rate,
                        momentum=config.momentum,
                        learning_rate_init=config.learning_rate_init)

    # fit and predict
    mlpr.fit(X_train, y_train.ravel())
    y_pred = mlpr.predict(X_test)

    # print(mlp_classification.score(X_test, y_test))

    wandb.sklearn.plot_regressor(mlpr, X_train, X_test, y_train, y_test, model_name='MLPRegressor')

    # log data to wandb
    #wandb.log({"conf_mat": wandb.plot.confusion_matrix(y_true=y_test.ravel(), preds=y_pred.ravel())})
    #wandb.log({"loss": zero_one_loss(y_test, y_pred)})
    #wandb.log({'accuracy': accuracy_score(y_test, y_pred)})
