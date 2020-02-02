"""
    File 'data_import.py' used to load and preprocess data for testing
        and prediction.
"""
import pandas as pd
import sklearn.preprocessing
import plotting


def data_loading():
    """
        Method to load data for training and prediction.
        return:
            1. training_data - pandas DataFrame (200000, 83) for model
                training
            2. data_to_predict - pandas DataFrame (100000, 79) that need to
                be predicted
    """
    # Training data loading
    training_data = pd.read_csv("data/input/train_data_200k.csv",
                                delimiter=',', index_col=False,
                                encoding='utf8')
    training_data = training_data.set_index('Unnamed: 0')

    # Data for prediction loading
    data_to_predict = pd.read_csv("data/input/test_data_100k.csv",
                                  delimiter=',', index_col=False,
                                  encoding='utf8')
    data_to_predict = data_to_predict.set_index('Unnamed: 0')

    return training_data, data_to_predict


def missing_values_filling(data):
    """
        Method to fill missing values using observations and linear
            interpolation.
        param:
            data - pandas DataFrame of data
        return:
            1. data_observations - pandas DataFrame of data filled using
                observations
            2. data_linear - pandas DataFrame of data filled using linear
                interpolation
    """
    # Missing values filling with observations
    data_observations = data.fillna(method='pad')
    data_observations = data_observations.fillna(method='bfill')

    # Missing values filling using linear interpolation
    data_linear = data.interpolate(method='linear',
                                   limit_direction='both')

    return data_observations, data_linear


def nan_columns_dropping(training_data, data_to_predict, folder):
    """
        Method to drop columns that can not be filled.
        param:
            1. training_data - pandas DataFrame of training data
            2. data_to_predict - pandas DataFrame of data that need to be
                predicted
            3. folder - string name of folder, where data need to be saved
        return:
            1. training_data - pandas DataFrame for training with dropped
                columns
            2. data_to_predict - pandas dataFrame that need to be predicted
                with dropped columns
    """
    # Columns that are full NaN
    full_nan_column_names = \
        data_to_predict.columns[data_to_predict.isna().any()].tolist()

    # Drop NaN columns from training data
    training_data = \
        training_data.drop(columns=full_nan_column_names)

    # Drop NaN columns from data for prediction
    data_to_predict = \
        data_to_predict.drop(columns=full_nan_column_names)

    # Save columns that are dropped
    dropped_columns = pd.DataFrame(full_nan_column_names, columns=['column'])
    dropped_columns.to_csv(path_or_buf="data/output/" + folder +
                                       "/sets/dropped_columns.csv")

    return training_data, data_to_predict


def sets_creation(data, folder,
                  to_predict=False):
    """
        Method to scale data and create testing and training sets for model.
        param:
            1. data - pandas DataFrame of data for training or prediction
            2. folder - string name of folder, where data need to be saved
            3. to_predict - boolean value of data for prediction or for
                training (False as default)
        return:
            1. If to_predict == True:
                1.1. data - pandas DataFrame of scaled data that need to be
                    predicted
                1.2. scale_model - sklearn MinMaxScaler model
            2. If to_predict == False:
                2.1. training_data - pandas DataFrame of training data
                2.2. testing_data - pandas DataFrame of testing data
    """
    # Get index and column names
    index_names = data.index.values
    column_names = data.columns.values

    # Create scaler model
    scale_model = sklearn.preprocessing.MinMaxScaler()

    if to_predict:
        # Create and train dummy data for scaler model
        data_dummy = pd.concat([data, data.iloc[:, :4]], axis=1)
        data_dummy = scale_model.fit_transform(data_dummy)

        data = data_dummy[:, :-4]

        # Save scaled data for prediction
        data = pd.DataFrame(data, columns=column_names, index=index_names)
        data.to_csv(path_or_buf="data/output/" + folder +
                                "/sets/data_to_predict.csv")

        return data, scale_model

    # Train and transform scaler model using data for traing
    data = scale_model.fit_transform(data)

    data = pd.DataFrame(data, columns=column_names, index=index_names)

    # Take 20% of traing data as testing sets
    testing_data = data.iloc[::5, :]
    training_data = data.drop(testing_data.index)

    # Save traing and testing sets
    testing_data.to_csv(path_or_buf="data/output/" + folder +
                                    "/sets/test_data.csv")
    training_data.to_csv(path_or_buf="data/output/" + folder +
                                     "/sets/train_data.csv")

    return training_data, testing_data


def data_preprocessing():
    """
        Method to preprocess data by creating training, testing sets and
            sets for prediction.
        return:
            1. data_observations_sets - list of training and testing
                sets which missing values are filled using observations
            2. data_to_predict_observations_with_scaler - list of scaled set
                filled using observations for prediction and its scaler model
            3. data_linear_sets - list of training and testing sets which
                missing values are filled using linear interpolation
            4. data_to_predict_linear_with_scaler - list of scaled set
                filled using linear regrression for prediction and its
                scaler model
            5. data_to_predict - pandas DataFrame of pure data for prediction
    """
    # Load data
    training_data, data_to_predict = data_loading()

    # Plot missing values
    plotting.plot_heatmap(training_data, "Missing data for training")
    plotting.plot_heatmap(data_to_predict, "Missing data for testing")

    # Fill missing data
    training_data_observations, training_data_linear = \
        missing_values_filling(training_data)
    data_to_predict_observations, data_to_predict_linear = \
        missing_values_filling(data_to_predict)

    # Drop columns that are NaN
    training_data_observations, data_to_predict_observations = \
        nan_columns_dropping(training_data_observations,
                             data_to_predict_observations,
                             "filled_by_observations")
    training_data_linear, data_to_predict_linear = \
        nan_columns_dropping(training_data_linear,
                             data_to_predict_linear,
                             "filled_linear")

    # Create training and testing sets
    data_observations_sets = \
        sets_creation(training_data_observations, "filled_by_observations")
    data_linear_sets = sets_creation(training_data_linear, "filled_linear")

    # Create sets for prediction
    data_to_predict_observations_with_scaler = \
        sets_creation(data_to_predict_observations, "filled_by_observations",
                      to_predict=True)
    data_to_predict_linear_with_scaler = \
        sets_creation(data_to_predict_linear, "filled_linear", to_predict=True)

    return data_observations_sets, data_to_predict_observations_with_scaler, \
        data_linear_sets, data_to_predict_linear_with_scaler, data_to_predict
