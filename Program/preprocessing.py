import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_data(
    df: pd.DataFrame,
    target_column: str,
    drop_columns: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42
):

    if drop_columns:
        df = df.drop(columns=drop_columns)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler