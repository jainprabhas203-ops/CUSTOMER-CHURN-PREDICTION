import pandas as pd
import numpy as np
import logging
import os
from utils import clean_column_names, handle_missing_values

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path="data/churn.csv"):
    """
    Load data from CSV file or generate synthetic data if file not found.
    """
    try:
        # Check if file exists
        if os.path.exists(file_path):
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
        else:
            logger.warning(f"Data file {file_path} not found. Generating synthetic data.")
            df = generate_synthetic_data()
        
        # Clean column names
        df = clean_column_names(df)
        
        # Handle missing values
        df = handle_missing_values(df)
        
        # Ensure target column exists
        if 'Churn' not in df.columns:
            # Try to find a similar column
            churn_cols = [col for col in df.columns if 'churn' in col.lower()]
            if churn_cols:
                df.rename(columns={churn_cols[0]: 'Churn'}, inplace=True)
            else:
                # Add a synthetic churn column if none exists
                df['Churn'] = np.random.choice(['Yes', 'No'], size=len(df), p=[0.25, 0.75])
        
        logger.info(f"Successfully loaded data with shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        # Return None to indicate failure
        return None

def generate_synthetic_data(n_samples=10000):
    """
    Generate synthetic customer churn data for demonstration purposes.
    """
    try:
        np.random.seed(42)  # For reproducibility
        
        # Generate features
        data = {
            'customerID': [f'ID{i:05d}' for i in range(n_samples)],
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'Partner': np.random.choice(['Yes', 'No'], n_samples, p=[0.52, 0.48]),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.30, 0.70]),
            'tenure': np.random.exponential(scale=20, size=n_samples).astype(int),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples, p=[0.45, 0.45, 0.1]),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.35, 0.40, 0.25]),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.30, 0.30, 0.40]),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.30, 0.30, 0.40]),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.30, 0.30, 0.40]),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.30, 0.30, 0.40]),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.30, 0.30, 0.40]),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.30, 0.30, 0.40]),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.50, 0.25, 0.25]),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples, p=[0.60, 0.40]),
            'PaymentMethod': np.random.choice([
                'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
            ], n_samples, p=[0.35, 0.25, 0.20, 0.20]),
            'MonthlyCharges': np.random.normal(64.80, 30.0, n_samples),
            'TotalCharges': np.random.normal(2280.0, 1500.0, n_samples)
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Ensure positive values for charges
        df['MonthlyCharges'] = np.abs(df['MonthlyCharges'])
        df['TotalCharges'] = np.abs(df['TotalCharges'])
        
        # Ensure tenure is positive
        df['tenure'] = np.abs(df['tenure'])
        
        # Generate churn based on some logic
        # Higher probability of churn for month-to-month contracts, higher monthly charges, shorter tenure
        churn_prob = (
            (df['Contract'] == 'Month-to-month').astype(int) * 0.3 +
            (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int) * 0.2 +
            (df['tenure'] < df['tenure'].median()).astype(int) * 0.3 +
            np.random.normal(0, 0.1, n_samples)  # Add some noise
        )
        
        # Cap probabilities between 0 and 1
        churn_prob = np.clip(churn_prob, 0, 1)
        
        # Generate churn labels
        df['Churn'] = np.random.binomial(1, churn_prob).astype(str)
        df['Churn'] = df['Churn'].replace({'0': 'No', '1': 'Yes'})
        
        logger.info(f"Generated synthetic data with shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error generating synthetic data: {e}")
        # Return an empty DataFrame as fallback
        return pd.DataFrame()

def sample_data(df, sample_size=5000):
    """
    Sample data for performance optimization in visualizations.
    """
    try:
        if len(df) <= sample_size:
            return df
        else:
            return df.sample(n=sample_size, random_state=42)
    except Exception as e:
        logger.error(f"Error sampling data: {e}")
        return df

def get_data_summary(df):
    """
    Get a summary of the data for display in the UI.
    """
    try:
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'missing_values': df.isnull().sum().to_dict(),
            'churn_rate': (df['Churn'] == 'Yes').mean() if 'Churn' in df.columns else 0
        }
        return summary
    except Exception as e:
        logger.error(f"Error getting data summary: {e}")
        return {}