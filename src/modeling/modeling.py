import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor


def optimize_hyper_paramters(model, parameters, train_x, train_y, scoring=None, folds=5):
    reg = GridSearchCV(model, param_grid=parameters, cv=folds,n_jobs=-1, scoring=scoring)
    reg.fit(train_x, train_y)

    best_mod = reg.best_estimator_
    print(reg.best_score_)
    print(reg.best_params_)

    return best_mod


def run_model(reg_mod, repeat=5):
    met_df = pd.DataFrame(columns=['aape', 'mape'])
    percent_df = pd.DataFrame(columns=[5, 10, 20])

    for i in range(repeat):
        X_train, y_train, holdout_x, holdout_y = get_data_split()
        reg_mod.fit(X_train.values, y_train.values)
        met, percent = evaluate_mod(reg_mod, holdout_x.values, holdout_y, features=X.columns)

        met_df = met_df.append(met, ignore_index=True)
        percent_df = percent_df.append(percent, ignore_index=True)

    return met_df, percent_df


def custom_model_scorer(y_true, y_pred):
    errors = np.abs((y_true - y_pred) / y_true)
    aape = np.mean(errors)

    return aape