"""
Name:       Model for multidimensional prediction
Purpose:    Create feedforward neural network to predict data
            that are filled with:
                1. Linear interpolation method
                2. Previous and next observations
Author:     Artem "Manneq" Arkhipov
Created:    02/02/2020
"""
import time
import data_import
import neural_network


"""
    File 'main.py' is main file that controls the sequence of function calls.
"""


def importing_data():
    """
        Method to import data.
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
    data_observations_sets, data_to_predict_observations_with_scaler, \
        data_linear_sets, data_to_predict_linear_with_scaler, \
        data_to_predict = data_import.data_preprocessing()

    return data_observations_sets, data_to_predict_observations_with_scaler, \
        data_linear_sets, data_to_predict_linear_with_scaler, data_to_predict


def predictions_making(data_sets, data_to_predict_with_scaler, data_to_predict,
                       folder):
    """
        Method make predictions using neural network.
        param:
            1. data_sets - list of training and testing sets
            2. data_to_predict_with_scaler - list of scaled set for
                prediction and its scaler model
            3. data_to_predict - pandas DataFrame of pure data for prediction
            4. folder - string name of folder, where data need to be saved
    """
    # Get names of columns that are need to be predicted
    target_names = data_sets[0].columns[77:].tolist()

    neural_network. neural_network_model(data_sets[0], data_sets[1],
                                         data_to_predict_with_scaler[0],
                                         data_to_predict_with_scaler[1],
                                         target_names, data_to_predict,
                                         folder)

    return


def main():
    """
        Main function.
    """
    # Data import
    print("========== Importing data ==========")
    time_start = time.time()

    data_observations_sets, data_to_predict_observations_with_scaler, \
        data_linear_sets, data_to_predict_linear_with_scaler, \
        data_to_predict = importing_data()

    time_end = (time.time() - time_start) / 60
    print("Data folder: \n\t    data/output/filled_by_observations/sets/"
          "\n\t    data/output/filled_linear/sets/")
    print("Image folder: plots/missing_data/")
    print("Done. With time " + str(time_end) + " min\n")

    # Data prediction using neural network,
    # that are trained on data filled by observations
    print("========== Making predictions using neural "
          "network for data filled by observations ==========")
    time_start = time.time()

    predictions_making(data_observations_sets,
                       data_to_predict_observations_with_scaler,
                       data_to_predict, "filled_by_observations")

    time_end = (time.time() - time_start) / 60
    print("Logs folder: data/output/filled_by_observations/"
          "neural_network/logs/")
    print("Weights folder: data/output/filled_by_observations/"
          "neural_network/weights/")
    print("Results folder: data/output/filled_by_observations/"
          "neural_network/result/")
    print("Done. With time " + str(time_end) + " min\n")

    # Data prediction using neural network,
    # that are trained on data filled using linear interpolation
    print("========== Making predictions using neural "
          "network for data filled with linear interpolation ==========")
    time_start = time.time()

    predictions_making(data_linear_sets, 
                       data_to_predict_linear_with_scaler, 
                       data_to_predict, "filled_linear")

    time_end = (time.time() - time_start) / 60
    print("Logs folder: data/output/filled_linear/neural_network/logs/")
    print("Weights folder: data/output/filled_linear/neural_network/weights/")
    print("Results folder: data/output/filled_linear/neural_network/result/")
    print("Done. With time " + str(time_end) + " min\n")

    return


if __name__ == "__main__":
    main()
