"""
    File 'neural_network.py' has functions for creating? testing and
        predicting feedworward neural network model.
"""
import pandas as pd
import keras
import numpy as np


def model_training_and_evaluation(model, training_set, testing_set, folder):
    """
        Method to train and evaluate neural network model.
        param:
            1. model - keras neural network model
            2. training_set - pandas DataFrame of training data
            3. testing_set - pandas trained DataFrame of testing data
            4. folder - string name of folder, where data need to be saved
        return:
            model - keras neural network trained model
    """
    # Set training parameters
    batch_size = 512
    epochs = 100

    # Train model
    model.fit(x=training_set.iloc[:, :77],
              y=training_set.iloc[:, 77:],
              batch_size=batch_size,
              epochs=epochs,
              verbose=0,
              validation_data=(testing_set.iloc[:, :77],
                               testing_set.iloc[:, 77:]),
              shuffle=True,
              callbacks=[keras.callbacks.TerminateOnNaN(),
                         keras.callbacks.EarlyStopping(
                             min_delta=1e-3,
                             patience=10),
                         keras.callbacks.ReduceLROnPlateau(),
                         keras.callbacks.TensorBoard(
                             log_dir="data/output/" + folder +
                                     "/neural_network/logs",
                             batch_size=batch_size,
                             write_grads=True,
                             write_images=True)])

    # evaluate model
    print("Metrics results: ")
    print(model.evaluate(x=testing_set.iloc[:, :77],
                         y=testing_set.iloc[:, 77:],
                         batch_size=batch_size,
                         verbose=0))

    # Save weights
    model.save_weights("data/output/" + folder +
                       "/neural_network/weights/weights.h5")

    return model


def prediction_making(model, prediction_set, scale_model, target_names,
                      data_to_predict, folder):
    """
        Method to make predictions using trained neural network model.
        param:
            1. model - keras neural network model
            2. prediction_set - set for values prediction
            3. scale_model - sklearn MinMaxScaler mode
            4. target_names - list of column names to predict
            5. data_to_predict - pandas DataFrame of data to predict
            6. folder - string name of folder, where data need to be saved
        return:
    """
    # Make prediction
    prediction_results = model.predict(prediction_set)

    # Apply reverse scaling on results
    prediction_results = scale_model.inverse_transform(
        np.concatenate((prediction_set, prediction_results), axis=1))[:, 77:]

    # Save results
    prediction_results = pd.DataFrame(prediction_results, columns=target_names,
                                      index=prediction_set.index.tolist())
    prediction_results = pd.concat([data_to_predict, prediction_results],
                                   axis=1)
    prediction_results.to_csv(path_or_buf="data/output/" + folder +
                                          "/neural_network/result/results.csv")

    return


def model_creation_and_compiling():
    """
        Method to create compile neural network model.
        return:
            model - keras neural network model
    """
    model = keras.Sequential()

    # Create model
    model.add(keras.layers.Dense(8192, input_dim=77, activation='relu'))
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dense(2048, activation='relu'))
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(8, activation='relu'))
    model.add(keras.layers.Dense(4, activation='sigmoid'))

    # Compile model
    model.compile(optimizer='adam', loss=['mean_squared_error'],
                  metrics=['accuracy', 'mse'])

    return model


def neural_network_model(training_set, testing_set, prediction_set,
                         scale_model, target_names, data_to_predict,
                         folder):
    """
        Method to use keras neural network model.
        param:
            1. training_set - pandas DataFrame of training data
            2. testing_set - pandas trained DataFrame of testing data
            3. prediction_set - set for values prediction
            4. scale_model - sklearn MinMaxScaler mode
            5. target_names - list of column names to predict
            6. data_to_predict - pandas DataFrame of data to predict
            7. folder - string name of folder, where data need to be saved
    """
    # Create and compile model
    model = model_creation_and_compiling()

    # Train and evaluate model
    model = model_training_and_evaluation(model, training_set,
                                          testing_set, folder)

    # Make a prediction
    prediction_making(model, prediction_set, scale_model, target_names,
                      data_to_predict, folder)

    return
