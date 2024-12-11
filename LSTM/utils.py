import numpy as np


def create_sequences(data, seq_length):
    """
    Create sequences for LSTM
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length, :])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)


def calculate_global_metrics(y_true, y_pred):
    """
    compute metrics
    """
    # Avoid division by zero for MAPE and SMAPE
    non_zero = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
    
    # RMSLE
    rmsle = np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred))**2))
    
    # MSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    return {"MAPE": mape, "SMAPE": smape, "RMSLE": rmsle, "RMSE": rmse}