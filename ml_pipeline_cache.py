import os
import joblib
import json
import logging
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = Path("models")
CACHE_DIR.mkdir(exist_ok=True)

def get_model_path(model_name):
    """Get the path for a cached model."""
    return CACHE_DIR / f"{model_name}.joblib"

def get_metadata_path(model_name):
    """Get the path for model metadata."""
    return CACHE_DIR / f"{model_name}_metadata.json"

def save_model(model_name, pipeline, feature_names, model, metadata=None):
    """
    Save a trained model pipeline with metadata.
    
    Args:
        model_name: Name identifier for the model (e.g., 'random_forest', 'xgboost', 'ensemble')
        pipeline: Trained sklearn pipeline
        feature_names: List of feature names after preprocessing
        model: The actual trained model (for feature importance)
        metadata: Additional metadata dictionary
    
    Returns:
        bool: True if save was successful
    """
    try:
        model_path = get_model_path(model_name)
        metadata_path = get_metadata_path(model_name)
        
        # Save the pipeline and feature names
        model_data = {
            'pipeline': pipeline,
            'feature_names': feature_names,
            'model': model
        }
        joblib.dump(model_data, model_path)
        
        # Save metadata
        meta = {
            'model_name': model_name,
            'training_date': datetime.now().isoformat(),
            'feature_count': len(feature_names) if feature_names else 0,
        }
        
        # Add custom metadata
        if metadata:
            meta.update(metadata)
        
        with open(metadata_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        logger.info(f"Model '{model_name}' saved successfully to {model_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving model '{model_name}': {e}")
        return False

def load_model(model_name):
    """
    Load a cached model pipeline.
    
    Args:
        model_name: Name identifier for the model
    
    Returns:
        dict: Dictionary with 'pipeline', 'feature_names', 'model', and 'metadata'
              or None if model doesn't exist or loading fails
    """
    try:
        model_path = get_model_path(model_name)
        metadata_path = get_metadata_path(model_name)
        
        # Check if model exists
        if not model_path.exists():
            logger.info(f"No cached model found for '{model_name}'")
            return None
        
        # Load the model data
        model_data = joblib.load(model_path)
        
        # Load metadata if it exists
        metadata = None
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        logger.info(f"Model '{model_name}' loaded successfully from cache")
        
        return {
            'pipeline': model_data['pipeline'],
            'feature_names': model_data['feature_names'],
            'model': model_data['model'],
            'metadata': metadata
        }
        
    except Exception as e:
        logger.error(f"Error loading model '{model_name}': {e}")
        return None

def model_exists(model_name):
    """Check if a cached model exists."""
    return get_model_path(model_name).exists()

def get_model_metadata(model_name):
    """Get metadata for a cached model."""
    try:
        metadata_path = get_metadata_path(model_name)
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        logger.error(f"Error reading metadata for '{model_name}': {e}")
        return None

def list_cached_models():
    """List all cached models with their metadata."""
    models = []
    try:
        for model_file in CACHE_DIR.glob("*.joblib"):
            model_name = model_file.stem
            metadata = get_model_metadata(model_name)
            models.append({
                'name': model_name,
                'path': str(model_file),
                'metadata': metadata
            })
    except Exception as e:
        logger.error(f"Error listing cached models: {e}")
    
    return models

def clear_cache(model_name=None):
    """
    Clear cached models.
    
    Args:
        model_name: Specific model to clear, or None to clear all
    
    Returns:
        bool: True if successful
    """
    try:
        if model_name:
            # Clear specific model
            model_path = get_model_path(model_name)
            metadata_path = get_metadata_path(model_name)
            
            if model_path.exists():
                model_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
            
            logger.info(f"Cleared cache for model '{model_name}'")
        else:
            # Clear all models
            for file in CACHE_DIR.glob("*"):
                file.unlink()
            logger.info("Cleared all cached models")
        
        return True
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return False

def get_cache_size():
    """Get the total size of cached models in MB."""
    try:
        total_size = sum(f.stat().st_size for f in CACHE_DIR.glob("*") if f.is_file())
        return total_size / (1024 * 1024)  # Convert to MB
    except Exception as e:
        logger.error(f"Error calculating cache size: {e}")
        return 0
