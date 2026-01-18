import pandas as pd


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Učitava CSV datoteku s energetskim podacima.

    :param csv_path: Putanja do CSV datoteke
    :return: Pandas DataFrame
    """
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Datoteka {csv_path} nije pronađena.")