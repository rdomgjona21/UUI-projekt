import pandas as pd
import numpy as np


def linear_regression_coefficients(model, feature_names, top_n=10):
    coefficients = model.coef_

    df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients,
        "AbsCoefficient": np.abs(coefficients)
    })

    df_sorted = df.sort_values("AbsCoefficient", ascending=False)
    return df_sorted.head(top_n)


def gradient_boosting_importance(model, feature_names, top_n=10):

    importances = model.feature_importances_

    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    df_sorted = df.sort_values("Importance", ascending=False)
    return df_sorted.head(top_n)