from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor


def get_models():
    """
    Vraća rječnik ML modela koji se koriste u radu.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "kNN": KNeighborsRegressor(n_neighbors=5),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }
    return models