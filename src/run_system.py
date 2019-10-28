import os

from src.data_preparation.data_utils import load_data
from src.pipelines.feature_creation import categorical_data_pipeline, new_features_pipeline, remove_unused_features
from src.pipelines.model_pipeline import model_selection, model_pipeline


ClEANED_DATA_FOLDER = '../data/cleaned/'
INTERMEDIATE_DATA_FOLDER = '../data/intermediate'

train_data_cleaned_path = os.path.join(ClEANED_DATA_FOLDER, 'train_data.csv')
test_data_cleaned_path = os.path.join(ClEANED_DATA_FOLDER, 'test_data.csv')

train_data_int_path = os.path.join(INTERMEDIATE_DATA_FOLDER, 'train_data.csv')
test_data_int_path = os.path.join(INTERMEDIATE_DATA_FOLDER, 'test_data.csv')


def run_system(selected_model):
    """ Run the complete pipeline """
    train_data = load_data(train_data_cleaned_path)
    test_data = load_data(test_data_cleaned_path)

    # Add new features
    train_data = new_features_pipeline(train_data)
    test_data = new_features_pipeline(test_data)

    # Handle Categorical Features
    train_data = categorical_data_pipeline(data=train_data, train_data=train_data)
    test_data = categorical_data_pipeline(data=test_data, train_data=train_data)

    # Dropping unused columns
    train_data = remove_unused_features(train_data)
    test_data = remove_unused_features(test_data)

    # Run model
    # Note: Model selection pipeline needs to run before function

    metrics_df = model_pipeline(train_data, selected_model)

    return metrics_df


def main():
    models = ['rf']
    rf_param = {'n_estimators': [50, 100, 150],
                'max_features': ['auto', 'sqrt'],
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 5, 10],
                'bootstrap': ['True', 'False']
                }

    parameters = [rf_param]

    train_data = load_data(train_data_cleaned_path)
    selected_model = model_selection(train_data=train_data, models=models, parameters=parameters)

    metrics = run_system(selected_model)

    print(metrics)


if __name__ == '__main__':
    main()


