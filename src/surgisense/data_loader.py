import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
