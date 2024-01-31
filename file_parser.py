import pandas as pd
import numpy as np


# Reads given CSV file and returns a pandas dataframe
def parse_csv(filepath: str, columns: list = None) -> pd.DataFrame:
    if columns is None:
        df = pd.read_csv(filepath)
    else:
        df = pd.read_csv(filepath, usecols=columns)
    return df
