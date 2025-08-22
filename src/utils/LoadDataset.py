import pandas as pd

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file and return as a DataFrame.
    
    :param file_path: Path to the CSV file.
    :return: DataFrame containing the dataset.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error