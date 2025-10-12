# scripts/utils.py

import pandas as pd
import os

def save_to_csv(dataframe, directory, filename):
    """
    Saves a pandas DataFrame to a CSV file.

    Args:
        dataframe (pd.DataFrame): The DataFrame to save.
        directory (str): The directory where the file will be saved.
        filename (str): The name of the output CSV file.
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = os.path.join(directory, filename)
    try:
        dataframe.to_csv(filepath, index=False)
        print(f"Successfully saved data to {filepath}")
    except Exception as e:
        print(f"Error saving data to {filepath}: {e}")

def save_to_json(dataframe, directory, filename):
    """
    Saves a pandas DataFrame to a JSON file.

    Args:
        dataframe (pd.DataFrame): The DataFrame to save.
        directory (str): The directory where the file will be saved.
        filename (str): The name of the output JSON file.
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = os.path.join(directory, filename)
    try:
        dataframe.to_json(filepath, orient='records', indent=4)
        print(f"Successfully saved data to {filepath}")
    except Exception as e:
        print(f"Error saving data to {filepath}: {e}")