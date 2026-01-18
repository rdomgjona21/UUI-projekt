from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def evaluate_model(model, X_test, y_test):
    """
    Evaluira model koristeći MAE i RMSE.
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return mae, rmse