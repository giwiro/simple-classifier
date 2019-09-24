import string

import numpy as np
import pandas as pd
from pandas import DataFrame

DEFAULT_LOCATION = "dataset.csv"
DEFAULT_TEST_LOCATION = "dataset_test.csv"


def load_dataset_pd(path: str = DEFAULT_LOCATION) -> DataFrame:
    return pd.read_csv(path, quotechar="\"")


def load_test_dataset_pd(path: str = DEFAULT_TEST_LOCATION) -> DataFrame:
    return pd.read_csv(path, quotechar="\"")


def get_categories():
    return np.array(["graphic cards"])


def normalize_text(name: str) -> str:
    # Here we need to get rid of all unsafe chars like emojis
    new_name = name.encode('ascii', 'ignore').decode('ascii')
    # Remove punctuation
    new_name = ''.join(c for c in new_name if c not in string.punctuation)
    # Remove numbers
    new_name = ''.join(c for c in new_name if c not in '0123456789')
    # Remove unnecessary white spaces
    new_name = ' '.join(new_name.split())
    # To lower case
    new_name = new_name.lower()
    return new_name
