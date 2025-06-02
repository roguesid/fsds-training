from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


def evaluate_model(model, X, y_true):
    y_pred = model.predict(X)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return {"mse": mse, "rmse": rmse, "mae": mae}
