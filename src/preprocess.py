import pandas as pd

def preprocess_data(df):
    # Select numeric columns only (simple start)
    df = df.select_dtypes(include=["number"])

    # Fill missing values
    df = df.fillna(df.mean())

    return df
