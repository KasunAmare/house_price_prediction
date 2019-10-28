from src.modeling.feature_engineering import *


def categorical_data_pipeline(data, train_data):
    data = add_dummies_categorical(data=data, train_data=train_data, feature='ViewType')
    data = target_encode_categorical(data=data,
                                     train_data=train_data,
                                     feature='ZoneCodeCounty',
                                     target_feature='SaleDollarCnt')

    return data


def new_features_pipeline(cleaned_data):
    cleaned_data = add_house_age(cleaned_data)

    cleaned_data = add_transaction_age(cleaned_data)

    residential_codes = ['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
    cleaned_data = add_residential(cleaned_data, residential_codes)

    cleaned_data = add_has_garage(cleaned_data)

    cleaned_data = add_has_view(cleaned_data)

    cleaned_data = add_distance_to_center(cleaned_data)

    return cleaned_data


def remove_unused_features(data):
    drop_columns = ['PropertyID',
                    'TransDate',
                    'censusblockgroup',
                    'Usecode',
                    'BGMedYearBuilt']

    data = data.drop(columns=drop_columns)

    return data




