import pandas as pd
from sklearn.model_selection import train_test_split

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lowercase with underscores."""
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

def split_data(df: pd.DataFrame, target_column: str, test_size=0.2, random_state=42):
    """Split the dataframe into train and test sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test



from sklearn.impute import SimpleImputer

def impute_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Fill missing values in numeric columns using specified strategy.
    
    Parameters:
    - df: Input DataFrame
    - strategy: Imputation strategy ('mean', 'median', or 'most_frequent')
    
    Returns:
    - DataFrame with imputed values
    """
    numeric_cols = df.select_dtypes(include='number').columns
    imputer = SimpleImputer(strategy=strategy)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df

from sklearn.preprocessing import LabelEncoder

def encode_categoricals(df: pd.DataFrame, method: str = 'label') -> pd.DataFrame:
    """
    Encode categorical (object) columns using label encoding or one-hot encoding.
    
    Parameters:
    - df: Input DataFrame
    - method: 'label' or 'onehot'
    
    Returns:
    - DataFrame with encoded categorical features
    """
    if method == 'label':
        for col in df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    elif method == 'onehot':
        df = pd.get_dummies(df)
    return df

from sklearn.preprocessing import StandardScaler, MinMaxScaler

def scale_features(df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
    """
    Scale numeric features using StandardScaler or MinMaxScaler.
    
    Parameters:
    - df: Input DataFrame
    - method: 'standard' or 'minmax'
    
    Returns:
    - DataFrame with scaled numeric features
    """
    scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

from scipy.stats import zscore
import numpy as np

def handle_outliers(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """
    Remove rows with outliers based on Z-score threshold.
    
    Parameters:
    - df: Input DataFrame
    - threshold: Z-score cutoff (default = 3.0)
    
    Returns:
    - DataFrame with outliers removed
    """
    numeric_cols = df.select_dtypes(include='number').columns
    z_scores = np.abs(zscore(df[numeric_cols]))
    return df[(z_scores < threshold).all(axis=1)]
