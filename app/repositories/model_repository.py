"""
Model Repository
Handles model loading and persistence following repository pattern
"""

import pickle
import os
from typing import Optional, Dict, Any

class ModelRepository:
    """Repository for model data access"""
    
    def __init__(self, model_path: str = 'best_model.pkl'):
        self.model_path = model_path
        self._model = None
        self._vectorizer = None
        self._normalizer = None
        self._model_name = None
    
    def load(self) -> bool:
        """
        Load model from file
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if not os.path.exists(self.model_path):
            return False
        
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self._model = model_data.get('model')
            self._vectorizer = model_data.get('vectorizer')
            self._normalizer = model_data.get('normalizer')
            self._model_name = model_data.get('model_name', 'Unknown')
            
            return self._model is not None and self._vectorizer is not None
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def get_model(self):
        """Get the loaded model"""
        return self._model
    
    def get_vectorizer(self):
        """Get the loaded vectorizer"""
        return self._vectorizer
    
    def get_normalizer(self):
        """Get the loaded normalizer"""
        return self._normalizer
    
    def get_model_name(self) -> Optional[str]:
        """Get the model name"""
        return self._model_name
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_name': self._model_name,
            'loaded': self.is_loaded(),
            'path': self.model_path
        }
