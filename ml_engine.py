import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import logging
import time

# Import caching module
from ml_pipeline_cache import save_model, load_model, model_exists

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_or_train_models(df, force_retrain=False):
    """
    Load cached models if available, otherwise train new models.
    
    Args:
        df: Training data DataFrame
        force_retrain: If True, ignore cache and retrain models
    
    Returns:
        Tuple of (pipelines, feature_names, models) or raises exception
    """
    # Model names for caching
    model_names = ['random_forest', 'xgboost', 'ensemble']
    
    # Check if all models are cached
    all_cached = all(model_exists(name) for name in model_names)
    
    if all_cached and not force_retrain:
        logger.info("Loading models from cache...")
        start_time = time.time()
        
        try:
            # Load all three models
            rf_data = load_model('random_forest')
            xgb_data = load_model('xgboost')
            ensemble_data = load_model('ensemble')
            
            if rf_data and xgb_data and ensemble_data:
                pipelines = (rf_data['pipeline'], xgb_data['pipeline'], ensemble_data['pipeline'])
                feature_names = (rf_data['feature_names'], xgb_data['feature_names'], ensemble_data['feature_names'])
                models = (rf_data['model'], xgb_data['model'], ensemble_data['model'])
                
                load_time = time.time() - start_time
                logger.info(f"Models loaded from cache in {load_time:.2f} seconds")
                
                return pipelines, feature_names, models
        except Exception as e:
            logger.warning(f"Error loading cached models: {e}. Will retrain.")
    
    # Train new models if cache not available or force_retrain is True
    logger.info("Training new models...")
    start_time = time.time()
    
    pipelines, feature_names, models = train_models(df)
    
    train_time = time.time() - start_time
    logger.info(f"Models trained in {train_time:.2f} seconds")
    
    # Save models to cache
    try:
        # Calculate accuracy for metadata
        X = df.drop('Churn', axis=1)
        y = (df['Churn'] == 'Yes').astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Save Random Forest
        rf_pred = pipelines[0].predict(X_test)
        rf_acc = accuracy_score(y_test, rf_pred)
        save_model('random_forest', pipelines[0], feature_names[0], models[0], {
            'accuracy': float(rf_acc),
            'training_time': train_time / 3,
            'data_shape': df.shape
        })
        
        # Save XGBoost
        xgb_pred = pipelines[1].predict(X_test)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        save_model('xgboost', pipelines[1], feature_names[1], models[1], {
            'accuracy': float(xgb_acc),
            'training_time': train_time / 3,
            'data_shape': df.shape
        })
        
        # Save Ensemble
        save_model('ensemble', pipelines[2], feature_names[2], models[2], {
            'training_time': train_time / 3,
            'data_shape': df.shape
        })
        
        logger.info("Models saved to cache successfully")
    except Exception as e:
        logger.warning(f"Error saving models to cache: {e}")
    
    return pipelines, feature_names, models


def train_models(df):
    """
    Train multiple ML models for churn prediction.
    Returns trained pipelines, feature columns, and models.
    """
    try:
        # Handle missing values in target column
        df = df.dropna(subset=['Churn'])
        
        # Prepare features and target
        X = df.drop('Churn', axis=1)
        y = (df['Churn'] == 'Yes').astype(int)  # Convert to binary
        
        # Identify numerical and categorical columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # Create preprocessing pipelines
        # Numerical preprocessing with imputation
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical preprocessing with imputation
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer([
            ('num', num_pipeline, numerical_cols),
            ('cat', cat_pipeline, categorical_cols)
        ])
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Random Forest Pipeline
        rf_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        ])
        
        # XGBoost Pipeline
        xgb_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', xgb.XGBClassifier(random_state=42, n_jobs=-1))
        ])
        
        # Ensemble Pipeline (Average of RF and XGBoost probabilities)
        ensemble_pipeline = Pipeline([
            ('preprocessor', preprocessor)
        ])
        
        # Train models
        rf_pipeline.fit(X_train, y_train)
        xgb_pipeline.fit(X_train, y_train)
        
        # For ensemble, we'll need to transform the data first
        X_train_transformed = ensemble_pipeline.named_steps['preprocessor'].fit_transform(X_train)
        X_test_transformed = ensemble_pipeline.named_steps['preprocessor'].transform(X_test)
        
        # Train individual models for ensemble
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_transformed, y_train)
        
        xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
        xgb_model.fit(X_train_transformed, y_train)
        
        # Store the trained models in the ensemble pipeline for later use
        ensemble_pipeline.rf_model = rf_model
        ensemble_pipeline.xgb_model = xgb_model
        
        # Evaluate models
        rf_pred = rf_pipeline.predict(X_test)
        xgb_pred = xgb_pipeline.predict(X_test)
        
        # Ensemble predictions (average probabilities)
        rf_proba = rf_model.predict_proba(X_test_transformed)[:, 1]
        xgb_proba = xgb_model.predict_proba(X_test_transformed)[:, 1]
        ensemble_proba = (rf_proba + xgb_proba) / 2
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        
        logger.info(f"Random Forest Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
        logger.info(f"XGBoost Accuracy: {accuracy_score(y_test, xgb_pred):.4f}")
        logger.info(f"Ensemble Accuracy: {accuracy_score(y_test, ensemble_pred):.4f}")
        
        # Get feature names after preprocessing
        rf_feature_names = get_feature_names_after_preprocessing(preprocessor, numerical_cols, categorical_cols)
        xgb_feature_names = rf_feature_names  # Same preprocessing
        ensemble_feature_names = rf_feature_names  # Same preprocessing
        
        # Return pipelines, feature columns, and trained models
        return (
            (rf_pipeline, xgb_pipeline, ensemble_pipeline),
            (rf_feature_names, xgb_feature_names, ensemble_feature_names),
            (rf_model, xgb_model, (rf_model, xgb_model))  # Individual models for feature importance
        )
        
    except Exception as e:
        logger.error(f"Error in train_models: {e}")
        raise

def get_feature_names_after_preprocessing(preprocessor, numerical_cols, categorical_cols):
    """
    Get feature names after preprocessing with ColumnTransformer.
    """
    try:
        # Get feature names from numerical transformer
        num_features = numerical_cols
        
        # Get feature names from categorical transformer
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features = cat_encoder.get_feature_names_out(categorical_cols)
        
        # Combine all feature names
        all_features = list(num_features) + list(cat_features)
        return all_features
    except Exception as e:
        logger.error(f"Error in get_feature_names_after_preprocessing: {e}")
        return []

def predict_churn(pipeline, X, model_name="Unknown", feature_columns=None):
    """
    Predict churn using the trained pipeline.
    """
    try:
        # Handle feature alignment if feature_columns is provided
        if feature_columns is not None:
            # Ensure X has the same columns as the model was trained on
            missing_cols = set(feature_columns) - set(X.columns)
            extra_cols = set(X.columns) - set(feature_columns)
            
            # Remove extra columns
            if extra_cols:
                X = X.drop(columns=extra_cols)
            
            # Add missing columns with default values
            for col in missing_cols:
                X[col] = 0  # Default value for missing columns
            
            # Reorder columns to match the training order
            X = X.reindex(columns=feature_columns, fill_value=0)
        
        # Special handling for ensemble model
        if model_name == "Ensemble":
            # Transform the data using the preprocessor
            X_transformed = pipeline.named_steps['preprocessor'].transform(X)
            # Get predictions from both models
            rf_proba = pipeline.rf_model.predict_proba(X_transformed)[:, 1]
            xgb_proba = pipeline.xgb_model.predict_proba(X_transformed)[:, 1]
            # Average the probabilities
            ensemble_proba = (rf_proba + xgb_proba) / 2
            # Convert to binary predictions
            predictions = (ensemble_proba > 0.5).astype(int)
        else:
            # Standard prediction for RF and XGBoost
            predictions = pipeline.predict(X)
        
        return predictions
    except Exception as e:
        logger.error(f"Error in predict_churn: {e}")
        # Return default predictions (all 0s) in case of error
        return np.zeros(len(X))

def get_feature_importance(model, feature_names, top_n=20):
    """
    Get feature importance from trained model.
    """
    try:
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models, take absolute values
            importances = np.abs(model.coef_[0])
        else:
            # Return None if model doesn't support feature importance
            return None
        
        # Create a DataFrame with feature names and importances
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        return feature_importance_df
    except Exception as e:
        logger.error(f"Error in get_feature_importance: {e}")
        return None
