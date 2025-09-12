# Assignment 2
# Reusable Python class
# Note: This class used to clean 3 datasets in the notebook named assignmet 2


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataCleaner:
    def __init__(self, strategy="mean", scaling="minmax", outlier_threshold=3):
        self.strategy = strategy
        self.scaling = scaling
        self.outlier_threshold = outlier_threshold
        self.logs = []

    def log(self, message):
        print(message)
        self.logs.append(message)
        
    def check_missing_values(self,df):
        numeric_cols = df.select_dtypes(include=[np.number])
        print (f'*** Total Missing Values = {numeric_cols.isnull().sum().sum()} ***')
                
    def missing_values(self,df):
        numeric_cols = df.select_dtypes(include=[np.number])
        return numeric_cols.isnull().sum().sum()
        

    def handle_missing(self, df):
        """Handle missing values"""
        for col in df.columns:
            num_of_missing_values = df[col].isnull().sum()
            if num_of_missing_values > 0:
                if self.strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
                    col_mean = df[col].mean()
                    
                    df[col] = df[col].replace(["nan", "NaN", "NA", ""], np.nan)
                    df[col] = df[col].fillna(col_mean)
                    #self.log(f"Filled  missing values in {col} with mean.")
                elif self.strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                    #self.log(f"Filled {num_of_missing_values} missing values in {col} with median.")
                elif self.strategy == "mode" and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mode()[0])
                    #self.log(f"Filled {num_of_missing_values} missing values in {col} with mode.")
                elif self.strategy == "drop":
                    df = df.dropna(subset=[col])
                    #self.log(f"Dropped rows with missing values in {col}.")
                else:
                    df[col] = df[col].fillna("unknown")
                    #self.log(f"Filled missing values in categorical {col} with 'unknown'.")
        self.log(f'Filled {self.missing_values(df)} missing values with {self.strategy}. ')
        return df

    def handle_outliers(self, df):
        """Remove outliers using z-score across all numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if df.empty or numeric_cols.empty:
            self.log("No numeric data for outlier detection.")
            return df

        mask = pd.Series(True, index=df.index)
        for col in numeric_cols:
            mean, std = df[col].mean(), df[col].std()
            if std == 0:
                continue
            z_scores = (df[col] - mean) / std
            col_mask = np.abs(z_scores) <= self.outlier_threshold
            mask &= col_mask
            outliers = np.sum(~col_mask)
            if outliers > 0:
                self.log(f"Detected {outliers} outliers in {col}.")
        df = df[mask]

        if df.empty:
            self.log("Warning: all rows dropped after outlier removal.")
        return df

    def scale_data(self, df):
        """Normalize or standardize all numeric data safely."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if df.empty or numeric_cols.empty:
            self.log("Skipping scaling (no numeric data or empty DataFrame).")
            return df

        if self.scaling == "standard":
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            self.log("Standardized numerical columns.")
        elif self.scaling == "minmax":
            scaler = MinMaxScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            self.log("Normalized numerical columns (Min-Max).")
        return df

    def scale_specific(self, df, columns, method="minmax"):
        """Scale only selected columns."""
        valid_cols = [col for col in columns if col in df.columns]
        if df.empty or not valid_cols:
            self.log("Skipping scale_specific (no valid columns or empty DataFrame).")
            return df

        if method == "standard":
            scaler = StandardScaler()
            df[valid_cols] = scaler.fit_transform(df[valid_cols])
            self.log(f"Standardized columns: {valid_cols}")
        elif method == "minmax":
            scaler = MinMaxScaler()
            df[valid_cols] = scaler.fit_transform(df[valid_cols])
            self.log(f"Normalized (Min-Max) columns: {valid_cols}")
        return df

    def clean_categorical(self, df):
        """Clean categorical text columns."""
        cat_cols = df.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].replace("nan", "unknown")
            self.log(f"Cleaned categorical column: {col}.")
        return df

    def clean(self,  dataset,df):
        """Run full cleaning pipeline."""
        self.logs = []
        df = df.copy()
        self.log(f"Starting cleaning process for {dataset}...")
        self.log("-------------------------------------------------")

        df = self.handle_missing(df)
        df = self.handle_outliers(df)
        df = self.scale_data(df)
        df = self.clean_categorical(df)

        self.log(f"Finished cleaning. Final shape: {df.shape}")
        return df, self.logs

# Note: This class used to clean 3 datasets in the notebook named assignmet 2