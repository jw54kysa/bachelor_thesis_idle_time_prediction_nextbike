import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import zero_one_loss, accuracy_score
from sklearn.model_selection import cross_validate, cross_val_score, GridSearchCV
from sklearn.neural_network import MLPClassifier
import wandb

idle_time_data = pd.read_csv('../data/df_points/df_points_18_21_class.csv')

TargetVariable = ['idle_time_class']
Predictors = ['bike_id', 'lat', 'lng', 'temp', 'rain', 'snow', 'dt_start', 'hex_enc', 'start_min', 'month', 'day']

X = idle_time_data[Predictors].values
y = idle_time_data[TargetVariable].values

PredictorScaler=StandardScaler()
PredictorScalerFit=PredictorScaler.fit(X)
X=PredictorScalerFit.transform(X)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, shuffle=False)
### shuffle/inorder train and test sets:

sweep_configuration = {
    "project": "MultiLayerPerceptronClassification",
    "name": "MLPC-sweep-scaled-1",
    "metric": {"name": "accuracy", "goal": "maximize"},
    "method": "random",
    "parameters": {
        "hidden_layer_sizes": {
            "values": [8, 16, 32, 64, 128, 256, 512]
        },
        "activation": {
            "values": ['identity', 'logistic', 'tanh', 'relu']
        },
        "solver": {
            "values": ['lbfgs','sgd','adam']
        },
        "alpha": {
            "values": [0.0001,0.0005,0.001, 0.005, 0.01, 0.05]
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

    _hidden_layer_sizes = wandb.config.hidden_layer_sizes
    _activation = wandb.config.activation
    _solver = wandb.config.solver
    _alpha = wandb.config.alpha
    _learning_rate = wandb.config.learning_rate
    _momentum = wandb.config.momentum

    model = MLPClassifier(hidden_layer_sizes=_hidden_layer_sizes,
                                   activation=_activation,
                                   solver=_solver,
                                   alpha=_alpha,
                                   learning_rate=_learning_rate,
                                   momentum=_momentum)

    model.fit(X_train,y_train.ravel())
    y_pred = model.predict(X_test)

    score_training = model.score(X_train, y_train.ravel())
    score_validation = model.score(X_test, y_test.ravel())

    acc = accuracy_score(y_test.ravel(), y_pred.ravel())
    loss = zero_one_loss(y_test.ravel(), y_pred.ravel())

    #wandb.sklearn.plot_feature_importances(model, Predictors)

    wandb.log({"accuracy": acc,"loss": loss})
    wandb.log({"conf_matrix": wandb.plot.confusion_matrix(y_true=y_test.ravel(), preds=y_pred.ravel())})
    #wandb.log({"feature_imp": wandb.sklearn.plot_feature_importances(model, Predictors)})
    wandb.log({"score_training":score_training, "score_validation":score_validation})

# INIT SWEEP
sweep_id_rfc = wandb.sweep(sweep_configuration, project="RandomForestClassifier")
# RUN SWEEP
wandb.agent(sweep_id_rfc, function=my_train_func)



