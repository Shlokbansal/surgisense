{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,














































































































































































































































































































































































































































































# notebooks/eda.ipynb

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('../data/post-operative-data.csv')  # adjust if needed
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Quick look
print(df.shape)
print(df.columns)
df.head()

# Summary
df.describe()
df.info()

# Target distribution (if labeled)
if 'complication' in df.columns:
    sns.countplot(x='complication', data=df)
    plt.title("Complication Label Distribution")
    plt.show()

 !pip install pandas numpy matplotlib seaborn scikit-learn shap jupyter

import sys
sys.path.append("../src")

from surgisense.data_loader import load_data
from surgisense.utils import clean_column_names, split_data

  # Load the data first
df = load_data("../data/post-operative-data.csv")

  # Clean column names
df = clean_column_names(df)

  # TEMPORARY target just for testing
df["dummy_target"] = 1  # Fake column for now
X_train, X_test, y_train, y_test = split_data(df, target_column="dummy_target")





import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()



# Debug information
print("DataFrame columns:", df.columns.tolist())
print("DataFrame dtypes:")
print(df.dtypes)

# Check which columns are numeric
numeric_cols = df.select_dtypes(include='number').columns
print("Numeric columns before dropping dummy_target:", numeric_cols.tolist())

# Make sure 'dummy_target' exists in numeric_cols before dropping
if 'dummy_target' in numeric_cols:
    numeric_cols = numeric_cols.drop('dummy_target')
else:
    print("Warning: 'dummy_target' not found in numeric columns!")

print("Final numeric columns:", numeric_cols.tolist())

# Check if we have any numeric columns to plot
if len(numeric_cols) > 0:
    df[numeric_cols].hist(
        bins=30,
        figsize=(16, 12),
        edgecolor='black',
        color='skyblue'
    )
    plt.suptitle("Distribution of Numeric Features (Excl. Dummy Target)", fontsize=18)
    plt.tight_layout()
    plt.show()
else:
    print("ERROR: No numeric columns found to plot!")
