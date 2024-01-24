import json 
import numpy as np
import pandas as pd 


def read_config(path):
    """Reads the config file and returns a dictionary."""
    try:
        with open(path) as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Config file not found.")
        config = None
    return config

def load_csv(path):
    """Loads the csv file and returns a dataframe."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print("CSV file not found.")
        df = None
    return df