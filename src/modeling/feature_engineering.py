import pandas as pd
import numpy as np
from geopy.distance import distance


# Categorical variables. Ensure that categorical features are handled at the end
# TODO: Ensure the same categories in train and test
def add_dummies_categorical(data, train_data, feature):
    data[feature] = data[feature].astype('category')
    train_data[feature] = train_data[feature].astype('category')

    temp_test = pd.get_dummies(data=data, columns=[feature])
    temp_train = pd.get_dummies(data=train_data, columns=[feature])

    missing_in_test = set(temp_train.columns) - set(temp_test.columns)
    new_in_test = set(temp_test.columns) - set(temp_train.columns)
    
    for c in missing_in_test:
        temp_test[c] = 0
    
    if new_in_test:
        temp_test = temp_test.drop(columns=new_in_test)     

    return temp_test


def target_encode_categorical(data, train_data, feature, target_feature):
    unique_vals = list(train_data[feature].unique())

    data['target_encoded_'+feature] = 0

    for val in unique_vals:
        msk = data[feature] == val
        msk_train = train_data[feature] == val
        data.loc[msk, 'target_encoded_'+feature] = train_data.loc[msk_train, target_feature].mean()

    data = data.drop([feature], axis=1)

    return data


# NOTE: This is a hardcoded function
def add_house_age(data):
    """
    Adds the age of the house as a feature and removes built year
    """
    data['HomeAge'] = pd.DatetimeIndex(data['TransDate']).year - data['BuiltYear']
    data = data.drop(['BuiltYear'], axis=1)

    return data


def add_price_per_sqft(data):
    """ The prices per square foot of the house and lot size
    :param data: all data
    :return: data with new features
    """

    data['BGSqFtPrice_lot'] = data['BGMedHomeValue'] / data['LotSizeSquareFeet']
    data['BGSqFtPrice_house'] = data['BGMedHomeValue'] / data['FinishedSquareFeet']

    return data


def add_transaction_age(data, ref_date='10/25/2019'):
    """ Indicate how old the transaction is, in terms of date
    :param data: data
    :param ref_date: the reference date which with respect to the counting is done
    :return: data with new feature
    """
    data['TransDate'] = pd.to_datetime(data['TransDate'])
    ref_date = pd.to_datetime(ref_date)
    data['DaysToTrans'] = (ref_date - data['TransDate'])/np.timedelta64(1, 'D')

    return data


def add_residential(data, residential_codes, zone_code_feature='ZoneCodeCounty'):
    """ Add a binary variable that indicate whether the house is in a residential area or not
    :param data: data
    :param residential_codes: the list of codes indicating a residential zonal code
    :param zone_code_feature: THe column name that has the zonal code
    :return: data with new feature
    """

    s = data[zone_code_feature]

    msk = (s.str.contains('|'.join(residential_codes)))

    data['residential'] = 0
    data.loc[msk, 'Residential'] = 1

    return data


def add_distance_to_center(data, center_lat_long=(47.608013, -122.335167)):
    """
    Indicate distance to the city center from the house. Uses the geopy package
    """

    lat = center_lat_long[0]
    long = center_lat_long[1]

    data['Latitude'] = data['Latitude']/1000000
    data['Longitude'] = data['Longitude'] / 1000000

    temp = data.apply(lambda x: distance((lat, long), (x['Latitude'], x['Longitude'])).km, axis=1)

    data['distance_to_center'] = temp

    return data


def add_has_garage(data):
    """
        Indicate whether the house has a garage
    """
    msk = data['GarageSquareFeet'] > 0

    data['HasGarage'] = 0
    data.loc[msk, 'HasGarage'] = 1

    return data


def add_has_view(data):
    """
        Indicate whether the house has a view.
    """
    msk = data['ViewType'] != 0

    data['HasView'] = 0
    data.loc[msk, 'HasView'] = 1

    return data




