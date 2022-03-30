import wandb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss
from sklearn.preprocessing import StandardScaler

WANDB_PROJECT_NAME = "rfc_hyperparam_opt"

with wandb.init(project=WANDB_PROJECT_NAME):
    # import data
    df = pd.read_csv('../../../data/df_points/df_points_18_21_class.csv')

    # define features and target
    TargetVariable = ['idle_time_class']
    Predictors = ['bike_id', 'lat', 'lng', 'temp', 'rain', 'snow', 'dt_start', 'hex_enc', 'start_min', 'month', 'day']
    X = df[Predictors].values
    y = df[TargetVariable].values

    # scale predictor features
    PredictorScaler = StandardScaler()
    PredictorScalerFit = PredictorScaler.fit(X)
    X = PredictorScalerFit.transform(X)

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # import sweep config
    config = wandb.config

    # define model
    rfc = RandomForestClassifier(
        criterion=config.criterion,
        bootstrap=config.bootstrap,
        max_depth=config.max_depth,
        max_features=config.max_features,
        min_samples_leaf=config.min_samples_leaf,
        min_samples_split=config.min_samples_split,
        n_estimators=config.n_estimators,
    )

    # fit and predict
    rfc.fit(X_train, y_train.ravel())
    y_pred = rfc.predict(X_test)

    # log data to wandb
    wandb.log({"conf_mat": wandb.plot.confusion_matrix(y_true=y_test.ravel(), preds=y_pred.ravel())})
    wandb.log({"feature_imp": wandb.sklearn.plot_feature_importances(rfc, Predictors)})
    wandb.log({"loss": zero_one_loss(y_test, y_pred)})
    wandb.log({'accuracy': accuracy_score(y_test, y_pred)})
