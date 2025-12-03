"""
Data Preprocessing Module for Credit Risk Assessment
Handles data cleaning, missing values, and outlier treatment
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Comprehensive data preprocessing for loan default prediction
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer_numeric = SimpleImputer(strategy='median')
        self.imputer_categorical = SimpleImputer(strategy='most_frequent')
        
    def load_data(self, filepath):
        """
        Load loan data from CSV
        
        Parameters:
        -----------
        filepath : str
            Path to the CSV file
            
        Returns:
        --------
        pd.DataFrame
            Loaded dataframe
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath, low_memory=False)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def handle_missing_values(self, df, threshold=0.5):
        """
        Handle missing values by dropping columns with high missing rate
        and imputing remaining ones
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        threshold : float
            Columns with missing rate > threshold will be dropped
            
        Returns:
        --------
        pd.DataFrame
            Processed dataframe
        """
        print("\nHandling missing values...")
        
        # Calculate missing percentages
        missing_pct = (df.isnull().sum() / len(df)).sort_values(ascending=False)
        
        # Drop columns with high missing rate
        cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
        print(f"Dropping {len(cols_to_drop)} columns with >{threshold*100}% missing data")
        df = df.drop(columns=cols_to_drop)
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Impute numeric columns
        if numeric_cols:
            df[numeric_cols] = self.imputer_numeric.fit_transform(df[numeric_cols])
        
        # Impute categorical columns
        if categorical_cols:
            df[categorical_cols] = self.imputer_categorical.fit_transform(df[categorical_cols])
        
        print(f"Missing values handled. Remaining shape: {df.shape}")
        return df
    
    def remove_outliers(self, df, columns, method='iqr', threshold=3):
        """
        Remove outliers using IQR method or Z-score
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        columns : list
            Columns to check for outliers
        method : str
            'iqr' or 'zscore'
        threshold : float
            Threshold for outlier detection
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with outliers removed
        """
        print(f"\nRemoving outliers using {method} method...")
        initial_rows = len(df)
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < threshold]
        
        removed_rows = initial_rows - len(df)
        print(f"Removed {removed_rows} rows ({removed_rows/initial_rows*100:.2f}%)")
        return df
    
    def encode_categorical(self, df, categorical_cols, method='label'):
        """
        Encode categorical variables
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        categorical_cols : list
            List of categorical columns to encode
        method : str
            'label' for Label Encoding, 'onehot' for One-Hot Encoding
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with encoded categorical variables
        """
        print(f"\nEncoding categorical variables using {method} encoding...")
        
        if method == 'label':
            for col in categorical_cols:
                if col not in df.columns:
                    continue
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                
        elif method == 'onehot':
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        print(f"Encoding complete. New shape: {df.shape}")
        return df
    
    def scale_features(self, df, columns_to_scale):
        """
        Scale numerical features using StandardScaler
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        columns_to_scale : list
            Columns to scale
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with scaled features
        """
        print("\nScaling numerical features...")
        
        # Only scale columns that exist
        cols = [col for col in columns_to_scale if col in df.columns]
        
        df[cols] = self.scaler.fit_transform(df[cols])
        print(f"Scaled {len(cols)} features")
        return df
    
    def prepare_target_variable(self, df, target_col='loan_status'):
        """
        Prepare target variable for binary classification
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_col : str
            Name of target column
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with binary target variable
        """
        print(f"\nPreparing target variable: {target_col}...")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        # Map loan status to binary (0 = Good, 1 = Default)
        default_statuses = ['Charged Off', 'Default', 'Late (31-120 days)', 
                           'Late (16-30 days)', 'Does not meet the credit policy. Status:Charged Off']
        
        df['default'] = df[target_col].apply(
            lambda x: 1 if x in default_statuses else 0
        )
        
        print(f"Target variable created:")
        print(f"  Good Loans (0): {(df['default']==0).sum()} ({(df['default']==0).sum()/len(df)*100:.2f}%)")
        print(f"  Defaulted (1): {(df['default']==1).sum()} ({(df['default']==1).sum()/len(df)*100:.2f}%)")
        
        return df
    
    def get_processed_data(self, df, target_col='default'):
        """
        Get final X and y for modeling
        
        Parameters:
        -----------
        df : pd.DataFrame
            Processed dataframe
        target_col : str
            Name of target column
            
        Returns:
        --------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target variable
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        print(f"\nFinal dataset prepared:")
        print(f"  Features: {X.shape[1]}")
        print(f"  Samples: {X.shape[0]}")
        
        return X, y


def preprocess_lending_club_data(filepath):
    """
    Complete preprocessing pipeline for Lending Club dataset
    
    Parameters:
    -----------
    filepath : str
        Path to raw data CSV
        
    Returns:
    --------
    X : pd.DataFrame
        Processed features
    y : pd.Series
        Target variable
    preprocessor : DataPreprocessor
        Fitted preprocessor object
    """
    preprocessor = DataPreprocessor()
    
    # Load data
    df = preprocessor.load_data(filepath)
    
    # Select relevant columns (modify based on your dataset)
    important_cols = [
        'loan_amnt', 'funded_amnt', 'term', 'int_rate', 'installment',
        'grade', 'sub_grade', 'emp_length', 'home_ownership', 'annual_inc',
        'verification_status', 'loan_status', 'purpose', 'dti', 'delinq_2yrs',
        'fico_range_low', 'fico_range_high', 'open_acc', 'pub_rec', 
        'revol_bal', 'revol_util', 'total_acc', 'mort_acc', 'pub_rec_bankruptcies'
    ]
    
    # Keep only columns that exist
    cols_to_keep = [col for col in important_cols if col in df.columns]
    df = df[cols_to_keep]
    
    # Prepare target variable first
    df = preprocessor.prepare_target_variable(df, target_col='loan_status')
    
    # Drop original loan_status column
    df = df.drop(columns=['loan_status'])
    
    # Handle missing values
    df = preprocessor.handle_missing_values(df, threshold=0.3)
    
    # Remove extreme outliers from key columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('default')  # Don't remove outliers from target
    df = preprocessor.remove_outliers(df, numeric_cols[:5], method='iqr')
    
    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = preprocessor.encode_categorical(df, categorical_cols, method='label')
    
    # Get X and y
    X, y = preprocessor.get_processed_data(df, target_col='default')
    
    # Scale features
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    X = preprocessor.scale_features(X, numeric_features)
    
    print("\n" + "="*50)
    print("PREPROCESSING COMPLETE!")
    print("="*50)
    
    return X, y, preprocessor


if __name__ == "__main__":
    # Example usage
    filepath = "data/raw/lending_club_loans.csv"
    X, y, preprocessor = preprocess_lending_club_data(filepath)
    
    # Save processed data
    X.to_csv('data/processed/X_processed.csv', index=False)
    y.to_csv('data/processed/y_processed.csv', index=False)
    print("\nProcessed data saved to data/processed/")
