import numpy as np
import pandas as pd

from src.modeling.modeling_utils import get_feature_importance_plot, get_accuracy_metrics
from sklearn.model_selection import GridSearchCV


def optimize_hyper_paramters(model, parameters, train_x, train_y, scoring=None, folds=5):
    """
    Run a grid search to find the best hyper parameters for a given model
    :param model: SKLearn model
    :param parameters: parameter grid to search
    :param train_x: train inputs
    :param train_y: train targets
    :param scoring: function that computes the score to optimize
    :param folds: number of folds for cross validation
    :return: the model with highest score
    """
    reg = GridSearchCV(model, param_grid=parameters, cv=folds,n_jobs=-1, scoring=scoring)
    reg.fit(train_x, train_y)

    best_mod = reg.best_estimator_
    print(reg.best_score_)
    print(reg.best_params_)

    return best_mod


def get_data_split(train_x, train_y, frac=0.20):
    """ Split the train set into two parts
    :param train_x: train inputs
    :param train_y: test inputs
    :param frac: portion held out for testing
    :return: train, test inputs and outputs
    """
    holdout_x = train_x.sample(frac=frac)
    holdout_indices = holdout_x.index
    X_train = train_x.drop(holdout_indices, axis=0)

    holdout_y = train_y.loc[holdout_indices]
    y_train = train_y.drop(holdout_indices, axis=0)

    return X_train, y_train, holdout_x, holdout_y


def evaluate_mod(model, test_x, test_y, features, plot=False):
    """ evaluate the model with a test set
    :param model: trained model
    :param test_x: test inputs
    :param test_y: test outputs
    :param features: feature names
    :param plot: to plot the feature importance or not
    :return:
    """
    y_hat = model.predict(test_x)
    res = get_accuracy_metrics(test_y, y_hat)

    if plot:
        get_feature_importance_plot(model, features)

    return res


def run_model(reg_mod, train_x, train_y, repeat=5):
    """ train and test the model and repeat to get accuracy scores for reporting
    :param reg_mod: model
    :param train_x: train_inputs
    :param train_y: train_outputs
    :param repeat: number of times to repeat
    :return: error scores and error distribution at 5%, 10%, 20%
    """
    metrics_df = pd.DataFrame(columns=['aape', 'mape'])

    for i in range(repeat):
        x_train, y_train, holdout_x, holdout_y = get_data_split(train_x, train_y)
        reg_mod.fit(x_train.values, y_train.values)
        met = evaluate_mod(reg_mod, holdout_x.values, holdout_y, features=train_x.columns)

        metrics_df = metrics_df.append(met, ignore_index=True)

    return metrics_df


def custom_model_scorer(y_true, y_pred):
    """ Calclate the AAPE for hyper parameter selection
    :param y_true: target
    :param y_pred: prediction
    :return: average absolute percent error
    """
    errors = np.abs((y_true - y_pred) / y_true)
    aape = np.mean(errors)

    return aape