import pandas as pd
import numpy as np
from scipy.stats import zscore

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# ───────────────────────────────────────────────
# Column Cleaning
# ───────────────────────────────────────────────
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lowercase with underscores."""
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

# ───────────────────────────────────────────────
# Train/Test Split
# ───────────────────────────────────────────────
def split_data(df: pd.DataFrame, target_column: str, test_size=0.2, random_state=42):
    """Split the dataframe into train and test sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

# ───────────────────────────────────────────────
# Missing Value Imputation
# ───────────────────────────────────────────────
def impute_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Fill missing values in numeric columns using specified strategy.
    """
    numeric_cols = df.select_dtypes(include='number').columns
    imputer = SimpleImputer(strategy=strategy)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df

# ───────────────────────────────────────────────
# Categorical Encoding
# ───────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame, method: str = 'label') -> pd.DataFrame:
    """
    Encode categorical (object) columns using label encoding or one-hot encoding.
    """
    df = df.copy()
    if method == 'label':
        for col in df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    elif method == 'onehot':
        df = pd.get_dummies(df)
    return df

# Feature Scaling
# ───────────────────────────────────────────────
def scale_features(df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
    """
    Scale numeric features using StandardScaler or MinMaxScaler.
    """
    df = df.copy()
    scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

# Outlier Handling
# ───────────────────────────────────────────────
def handle_outliers(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """
    Remove rows with outliers based on Z-score threshold.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) > 0:
        z_scores = np.abs(zscore(df[numeric_cols]))
        df = df[(z_scores < threshold).all(axis=1)]
    return df

# ───────────────────────────────────────────────
# Full Preprocessing Pipeline
# ────────────────────────
def run_full_pipeline(df: pd.DataFrame, encode_method="label", outlier_thresh=3.0) -> pd.DataFrame:
    """Run full data preprocessing pipeline on a raw DataFrame."""
    df = clean_column_names(df)

    # Encode categorical data first (this handles the medical categorical values)
    df = encode_categoricals(df, method=encode_method)
    
    # Now handle numeric operations
    df = handle_outliers(df, threshold=outlier_thresh)
    df = scale_features(df)

    return df