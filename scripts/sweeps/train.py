import wandb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

WANDB_PROJECT_NAME = "hyperparameter-optimization_class"

with wandb.init(project=WANDB_PROJECT_NAME):
    df = pd.read_csv('../../data/df_points/df_points_18_21_class.csv')

    TargetVariable = ['idle_time_class']
    Predictors = ['bike_id', 'lat', 'lng', 'temp', 'rain', 'snow', 'dt_start', 'hex_enc', 'start_min', 'month', 'day']

    X = df[Predictors].values
    y = df[TargetVariable].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    config = wandb.config
    rfc = RandomForestClassifier(
        bootstrap=config.bootstrap,
        max_depth = config.max_depth,
        max_features = config.max_features,
        min_samples_leaf = config.min_samples_leaf,
        min_samples_split = config.min_samples_split,
        n_estimators = config.n_estimators,
    )

    rfc.fit(X_train, y_train.ravel())
    y_pred = rfc.predict(X_test)

    labels = ["vs", "s", "l", "vl"]

    wandb.log({"conf_mat": wandb.plot.confusion_matrix(y_true=y_test.ravel(),preds=y_pred.ravel())})

    wandb.log({'accuracy': accuracy_score(y_test, y_pred)})
