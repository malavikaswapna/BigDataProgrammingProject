import numpy as np
import pandas as pd
import dask.dataframe as dd

# Load datasets
df_full = pd.read_csv("Trips_Full_Data.csv")
df = pd.read_csv("Trips_by_Distance.csv")

# Data Cleaning
# Convert 'Population staying at home' column to integers
df["Population Staying at Home"] = df["Population Staying at Home"].fillna(0).astype("int64")

# Display data types of columns
print("Data Types of Columns:")
print(df.dtypes)

# Count non-null values for each column
print("\nCount of Non-null Values for Each Column:")
print(df.notnull().sum())

# Count null values for each column
print("\nCount of Null Values for Each Column:")
print(df.isna().sum())

# Compute descriptive statistics for the whole dataframe
print("\nDescriptive Statistics for the Whole DataFrame:")
print(df.describe())

# Compute descriptive statistics for the 'Population staying at home' column
print("\nDescriptive Statistics for 'Population Staying at Home' Column:")
print(df['Population Staying at Home'].describe())
