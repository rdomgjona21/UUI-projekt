import pandas as pd
import numpy as np


def linear_regression_coefficients(model, feature_names, top_n=10):
    """
    Analiza koeficijenata linearne regresije.

    :param model: istrenirani LinearRegression model
    :param feature_names: nazivi ulaznih značajki
    :param top_n: broj najutjecajnijih značajki
    :return: DataFrame s najvažnijim koeficijentima
    """
    coefficients = model.coef_

    df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients,
        "AbsCoefficient": np.abs(coefficients)
    })

    df_sorted = df.sort_values("AbsCoefficient", ascending=False)
    return df_sorted.head(top_n)


def gradient_boosting_importance(model, feature_names, top_n=10):
    """
    Analiza važnosti značajki kod Gradient Boosting modela.

    :param model: istrenirani GradientBoostingRegressor
    :param feature_names: nazivi ulaznih značajki
    :param top_n: broj najvažnijih značajki
    :return: DataFrame s najvažnijim značajkama
    """
    importances = model.feature_importances_

    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    df_sorted = df.sort_values("Importance", ascending=False)
    return df_sorted.head(top_n)