from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor


def get_models():
    models = {
        "Linear Regression": LinearRegression(),
        "kNN": KNeighborsRegressor(n_neighbors=5),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }
    return models