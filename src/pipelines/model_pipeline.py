import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

from src.data_preparation.data_utils import split_input_output_data
from src.modeling.modeling import *
from src.modeling.modeling_utils import plot_error_distribution


def model_selection(train_data, models, parameters):
    """ finds the model that gives the lowest error
    :param train_data: train data
    :param models: list of models ( sklearn objects)
    :param parameters: list of dictionaries with parameter grids for each model
    :return: best model
    """

    models_dict = {'rf': RandomForestRegressor(),
                   'dt': DecisionTreeRegressor(),
                   'gb': GradientBoostingRegressor()}

    X, y = split_input_output_data(train_data, output_column='SaleDollarCnt')
    X_train, y_train, holdout_x, holdout_y = get_data_split(train_x=X, train_y=y, frac=0.20)

    best_aape = 1.00
    selected_mod = None

    for m in models:
        model = models_dict[m]
        best_mod = optimize_hyper_paramters(model, parameters, X_train, y_train, scoring=custom_model_scorer)
        best_mod.fit(X_train, y_train)

        results = evaluate_mod(best_mod, holdout_x.values, holdout_y, features=holdout_x.columns)

        if results['aape'] < best_aape:
            selected_mod = best_mod

    return selected_mod


def model_pipeline(train_data, model):
    X, y = split_input_output_data(train_data, output_column='SaleDollarCnt')

    metrics_df = run_model(reg_mod=model, train_x=X, train_y=y, repeat=5)

    # Get plots
    X_train, y_train, holdout_x, holdout_y = get_data_split(train_x=X, train_y=y, frac=0.20)
    model.fit(X_train.values, y_train.values)
    y_pred = model.predict(holdout_x)

    plot_error_distribution(y_pred, holdout_y)
    get_feature_importance_plot(model, features=X.columns)

    return metrics_df

