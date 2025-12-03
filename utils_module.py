"""
Utility Functions for Credit Risk Assessment Project
Helper functions used across the project
"""

import pandas as pd
import numpy as np
import yaml
import pickle
import json
import os
from datetime import datetime
import logging


def load_config(config_path='config.yaml'):
    """
    Load configuration from YAML file
    
    Parameters:
    -----------
    config_path : str
        Path to config file
        
    Returns:
    --------
    dict : Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_file='logs/credit_risk.log', level=logging.INFO):
    """
    Setup logging configuration
    
    Parameters:
    -----------
    log_file : str
        Path to log file
    level : int
        Logging level
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def create_directories(directories):
    """
    Create project directories if they don't exist
    
    Parameters:
    -----------
    directories : list
        List of directory paths to create
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        

def save_pickle(obj, filepath):
    """
    Save object as pickle file
    
    Parameters:
    -----------
    obj : object
        Object to save
    filepath : str
        Path to save file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Object saved to {filepath}")


def load_pickle(filepath):
    """
    Load pickle file
    
    Parameters:
    -----------
    filepath : str
        Path to pickle file
        
    Returns:
    --------
    object : Loaded object
    """
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    return obj


def save_json(data, filepath):
    """
    Save data as JSON file
    
    Parameters:
    -----------
    data : dict
        Data to save
    filepath : str
        Path to save file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"JSON saved to {filepath}")


def load_json(filepath):
    """
    Load JSON file
    
    Parameters:
    -----------
    filepath : str
        Path to JSON file
        
    Returns:
    --------
    dict : Loaded data
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def calculate_credit_score(fico_low, fico_high):
    """
    Calculate average credit score from FICO range
    
    Parameters:
    -----------
    fico_low : float
        Lower FICO score
    fico_high : float
        Upper FICO score
        
    Returns:
    --------
    float : Average credit score
    """
    return (fico_low + fico_high) / 2


def categorize_fico_score(fico_score):
    """
    Categorize FICO score into risk categories
    
    Parameters:
    -----------
    fico_score : float
        FICO credit score
        
    Returns:
    --------
    str : Risk category
    """
    if fico_score < 580:
        return "Poor"
    elif fico_score < 670:
        return "Fair"
    elif fico_score < 740:
        return "Good"
    elif fico_score < 800:
        return "Very Good"
    else:
        return "Excellent"


def calculate_debt_burden(installment, annual_income):
    """
    Calculate debt burden ratio
    
    Parameters:
    -----------
    installment : float
        Monthly installment amount
    annual_income : float
        Annual income
        
    Returns:
    --------
    float : Debt burden ratio
    """
    monthly_income = annual_income / 12
    if monthly_income == 0:
        return 0
    return installment / monthly_income


def calculate_loan_to_income(loan_amount, annual_income):
    """
    Calculate loan to income ratio
    
    Parameters:
    -----------
    loan_amount : float
        Loan amount
    annual_income : float
        Annual income
        
    Returns:
    --------
    float : Loan to income ratio
    """
    if annual_income == 0:
        return 0
    return loan_amount / annual_income


def print_section_header(title, width=70):
    """
    Print formatted section header
    
    Parameters:
    -----------
    title : str
        Section title
    width : int
        Width of header
    """
    print("\n" + "="*width)
    print(title.center(width))
    print("="*width + "\n")


def format_currency(amount):
    """
    Format number as currency
    
    Parameters:
    -----------
    amount : float
        Amount to format
        
    Returns:
    --------
    str : Formatted currency string
    """
    return f"${amount:,.2f}"


def format_percentage(value):
    """
    Format number as percentage
    
    Parameters:
    -----------
    value : float
        Value to format (0-1)
        
    Returns:
    --------
    str : Formatted percentage string
    """
    return f"{value*100:.2f}%"


def calculate_model_improvement(model_profit, baseline_profit):
    """
    Calculate improvement of model over baseline
    
    Parameters:
    -----------
    model_profit : float
        Profit with model
    baseline_profit : float
        Baseline profit
        
    Returns:
    --------
    tuple : (improvement_amount, improvement_percentage)
    """
    improvement = model_profit - baseline_profit
    if baseline_profit == 0:
        improvement_pct = 0
    else:
        improvement_pct = (improvement / abs(baseline_profit)) * 100
    return improvement, improvement_pct


def generate_timestamp():
    """
    Generate timestamp string for file naming
    
    Returns:
    --------
    str : Timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def validate_dataframe(df, required_columns):
    """
    Validate that dataframe contains required columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to validate
    required_columns : list
        List of required column names
        
    Returns:
    --------
    bool : True if valid, False otherwise
    """
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        return False
    return True


def get_feature_types(df):
    """
    Classify features into numerical and categorical
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    tuple : (numerical_columns, categorical_columns)
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    return numerical_cols, categorical_cols


def print_memory_usage(df):
    """
    Print memory usage of dataframe
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    """
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory usage: {memory_usage:.2f} MB")


def reduce_memory_usage(df):
    """
    Reduce memory usage by downcasting numeric types
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame : Optimized dataframe
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage().sum() / 1024**2
    reduction = 100 * (start_mem - end_mem) / start_mem
    
    print(f'Memory usage decreased from {start_mem:.2f}MB to {end_mem:.2f}MB ({reduction:.1f}% reduction)')
    
    return df


class Timer:
    """
    Context manager for timing code execution
    """
    def __init__(self, name="Operation"):
        self.name = name
        
    def __enter__(self):
        self.start = datetime.now()
        print(f"Starting {self.name}...")
        return self
        
    def __exit__(self, *args):
        self.end = datetime.now()
        self.duration = (self.end - self.start).total_seconds()
        print(f"{self.name} completed in {self.duration:.2f} seconds")


def create_risk_bins(probabilities, bins=5):
    """
    Create risk bins from default probabilities
    
    Parameters:
    -----------
    probabilities : array-like
        Default probabilities
    bins : int
        Number of bins
        
    Returns:
    --------
    array : Risk categories
    """
    return pd.cut(probabilities * 100, bins=bins, 
                 labels=[f'Risk_{i+1}' for i in range(bins)])


def print_project_info():
    """
    Print project information banner
    """
    banner = """
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║     Credit Risk Assessment & Loan Default Prediction  ║
    ║                                                       ║
    ║     ML Project for Business Analyst Portfolio        ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """
    print(banner)


# Example usage and testing
if __name__ == "__main__":
    print_project_info()
    
    # Test configuration loading
    try:
        config = load_config('config.yaml')
        print("✓ Configuration loaded successfully")
    except:
        print("✗ Configuration file not found")
    
    # Test directory creation
    directories = [
        'data/raw',
        'data/processed',
        'models/saved_models',
        'reports/figures',
        'logs'
    ]
    create_directories(directories)
    print("✓ Directories created successfully")
    
    # Test timing
    with Timer("Sample operation"):
        import time
        time.sleep(1)
    
    print("\n✅ Utility functions module working correctly!")
