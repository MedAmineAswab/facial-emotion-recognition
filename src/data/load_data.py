
import pandas as pd

def load_data(path):
    """
    Charge les données à partir d'un fichier CSV.

    Args:
        path (str): chemin du fichier CSV

    Returns:
        pd.DataFrame
    """
    df = pd.read_csv(path)

    print(f"Data loaded from {path}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    return df
  