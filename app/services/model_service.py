"""
Model Service
Business logic for model management
"""

from typing import Dict, Any
from app.repositories.model_repository import ModelRepository

class ModelService:
    """Service for handling model operations"""
    
    def __init__(self, model_repository: ModelRepository):
        self.model_repository = model_repository
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of the model
        
        Returns:
            Dict containing health status information
        """
        return {
            'status': 'healthy',
            'model_loaded': self.model_repository.is_loaded(),
            'model_info': self.model_repository.get_model_info()
        }
    
    def get_api_info(self) -> Dict[str, Any]:
        """
        Get API information
        
        Returns:
            Dict containing API information
        """
        model_name = self.model_repository.get_model_name()
        return {
            'message': 'SMS Spam Classification API',
            'status': 'running',
            'model': model_name if model_name else 'Not loaded',
            'endpoints': {
                '/predict': 'POST - Classify text (body: {"text": "your text here"})',
                '/predict_batch': 'POST - Batch predictions (body: {"texts": ["text1", "text2"]})',
                '/health': 'GET - Check API health',
                '/api': 'GET - API information'
            }
        }
