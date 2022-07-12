import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import zero_one_loss, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import wandb
import random

idle_time_data = pd.read_csv('../../data/final_df_points_18_21_class.csv')

def get_binary_class(row):
    if row['idle_time'] < 1439:
        val = '1day'
    else:
        val = 'longer'
    return val

idle_time_data['idle_time_class_binary'] = idle_time_data.apply(get_binary_class, axis=1)

TargetVariable = ['idle_time_class_binary']
Predictors = ['bike_id', 'lat', 'lng', 'temp', 'rain', 'snow', 'wind_speed', 'humidity', 'dt_start',
              'hex_enc', 'start_min', 'year', 'month', 'day', 'on_station', 'in_zone', 'zone_name_enc']


X = idle_time_data[Predictors].values
y = idle_time_data[TargetVariable].values

PredictorScaler = StandardScaler()
PredictorScalerFit = PredictorScaler.fit(X)
X = PredictorScalerFit.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, shuffle=False)

sweep_configuration = {
    "project": "MLP-Classification",
    "name": "MLPC-sweep-binary-1d",
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

def eval_classification(y_test,y_pred,labels):
    # Metrics
    # Accuracy, precision, recall
    acc = accuracy_score(y_test, y_pred.ravel())
    macro_precision = precision_score(y_test.ravel(), y_pred.ravel(), average='macro', labels=labels)
    micro_precision = precision_score(y_test.ravel(), y_pred.ravel(), average='micro', labels=labels)
    macro_recall = recall_score(y_test.ravel(), y_pred.ravel(), average='macro', labels=labels)
    micro_recall = recall_score(y_test.ravel(), y_pred.ravel(), average='micro', labels=labels)

    macro_f1 = f1_score(y_test.ravel(), y_pred.ravel(), average='macro', labels=labels)
    micro_f1 = f1_score(y_test.ravel(), y_pred.ravel(), average='micro', labels=labels)

    print(acc)
    print(macro_precision, micro_precision)
    print(macro_recall, micro_recall)
    print(macro_f1, micro_f1)

    return acc, macro_precision, micro_precision, macro_recall, micro_recall, macro_f1, micro_f1

def my_train_func():
    wandb.init()

    hls = [(16), (32), (64), (128),
           (16, 16), (32, 32), (64, 64), (128, 128),
           (16, 32, 16), (32, 64, 32), (64, 128, 64),
           (128, 64, 128)]

    _hidden_layer_sizes = random.choice(hls)
    print(_hidden_layer_sizes)
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

    acc, macro_precision, micro_precision, macro_recall, micro_recall, macro_f1, micro_f1 = eval_classification(y_test,y_pred,["< 1day","> 1day"])

    wandb.log({"accuracy": acc})
    wandb.log({"conf_matrix": wandb.plot.confusion_matrix(y_true=y_test.ravel(), preds=y_pred.ravel())})
    wandb.log({"macro_precision": macro_precision, "micro_precision": micro_precision})
    wandb.log({"macro_recall": macro_recall, "micro_recall": micro_recall})
    wandb.log({"macro_f1": macro_f1, "micro_f1": micro_f1})


# INIT SWEEP
sweep_id_rfc = wandb.sweep(sweep_configuration, project="MLP-Classification")
# RUN SWEEP
wandb.agent(sweep_id_rfc, function=my_train_func)
