import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Dense
import keras.backend as K
import numpy as np
from sklearn.metrics import root_mean_squared_error
import optuna

class NYCSalesLSTM(Model):
    def __init__(
            self, 
            units: int,
            dropout: float, 
            #sequence_length: int, 
            #num_features: int
    ):
        super(NYCSalesLSTM, self).__init__()
        self.lstm = LSTM(
            units=units, 
            activation='relu', 
            # input_shape=(sequence_length, num_features),
            return_sequences=False,
            dropout=dropout
        )
        self.dense = Dense(1)

    def call(self, inputs):
        x = self.lstm(inputs)
        return self.dense(x)
    
    def compile_and_train(
            self,
            X_train, 
            y_train, 
            X_val=None, 
            y_val=None, 
            batch_size=32, 
            epochs=100, 
            # patience=5, 
            verbose=2
    ):
        """Compile and train the model."""
        self.compile(optimizer='adam', loss='mse')

        '''
        # Early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )
        '''
        
        # Fit the model
        # To align with other model, here validation_data is commented
        history = self.fit(
            X_train, y_train,
            # validation_data=(X_val, y_val) if X_val is not None else None,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            # callbacks=[early_stopping]
        )
        return history
    
    def inverse_transform_prediction(self, X_train, X_test, y_train, y_test, scaler):
        num_features = X_train.shape[2]

        # Predict on the training data to visualize trend
        predicted_train = self.predict(X_train)
        predicted_test = self.predict(X_test)

        # Combine predicted and true values for inverse transformation
        all_predicted = np.vstack([predicted_train, predicted_test])
        all_actual = np.vstack([y_train.reshape(-1, 1), y_test.reshape(-1, 1)])

        # Inverse transform the predicted values
        all_predicted_original = scaler.inverse_transform(
            np.hstack([all_predicted, np.zeros((all_predicted.shape[0], num_features - 1))])
        )[:, 0]

        # Inverse transform the actual values
        all_actual_original = scaler.inverse_transform(
            np.hstack([all_actual, np.zeros((all_actual.shape[0], num_features - 1))])
        )[:, 0]

        # Split back into train and test
        predicted_train_original = all_predicted_original[:len(y_train)]
        predicted_test_original = all_predicted_original[len(y_train):]
        y_train_original = all_actual_original[:len(y_train)]
        y_test_original = all_actual_original[len(y_train):]

        # print
        mse_train = root_mean_squared_error(y_train, predicted_train)
        print(f"Final Training Loss (MSE): {mse_train}")

        # Calculate MSE and RMSE
        mse_test = root_mean_squared_error(y_test, predicted_test)
        print((f"Final Test MSE: {mse_test}"))

        return predicted_train_original, predicted_test_original, y_train_original, y_test_original

def rolling_validation(
        model_config,
        X, 
        y, 
        scaler,
        validation_size, 
        initial_train_ratio=0.6, 
        step_size=1,
    ):
    """
    Performs rolling validation on a time series dataset using a specified model class.
    """
    # Determine the initial training size
    initial_train_size = int(len(X) * initial_train_ratio)
    val_predictions = []
    errors_val = []
    errors_train = []


    # Rolling splits
    split_idx = initial_train_size
    splits_done = 0
    while split_idx + validation_size <= len(X):
        # Define training and validation sets
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_val = X[split_idx:split_idx + validation_size]
        y_val = y[split_idx:split_idx + validation_size]

        model = NYCSalesLSTM(
            units=model_config["units"],
            dropout=model_config["dropout"],
            # sequence_length=X.shape[1],
            # num_features=X.shape[2]
        )

        model.compile_and_train(
            X_train=X_train, 
            y_train=y_train, 
            X_val=X_val, 
            y_val=y_val, 
            batch_size=model_config["batch_size"], 
            epochs=model_config["epoches"], 
            # patience=model_config["patience"], 
            verbose=model_config["verbose"]
        )
        
        # Predict and inverse transform
        predicted_train_original, predicted_test_original, y_train_original, y_test_original = model.inverse_transform_prediction(
            X_train, 
            X_val, 
            y_train, 
            y_val, 
            scaler
        )
        val_predictions.append(predicted_test_original)

        # evaluate
        rmse_train = root_mean_squared_error(y_train_original, predicted_train_original)
        errors_train.append(rmse_train)

        rmse_val = root_mean_squared_error(y_test_original, predicted_test_original)
        errors_val.append(rmse_val)

        
        # Expand the training set
        split_idx += step_size
        splits_done += 1
        
    return errors_train, errors_val, val_predictions

def objective(trial, X, y, scaler):
    """Objective function for hyper-param search via optuna"""
    # Suggest hyperparameters
    model_config = {
        "units": trial.suggest_categorical("units", [32, 64, 128]),
        "dropout": trial.suggest_float("dropout", 0.05, 0.15),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "epoches": trial.suggest_categorical("epoches", [50, 75, 100]),
        "patience": 5,
        "verbose": 0
    }
    
    errors_train, errors_val, val_predictions = rolling_validation(
        model_config=model_config,
        X=X,
        y=y,
        scaler=scaler,
        validation_size=1,
        initial_train_ratio=0.6,
        step_size=1
    )

    # Store val_predictions in the trial
    trial.set_user_attr("val_predictions", val_predictions)
    
    return np.mean(errors_val)