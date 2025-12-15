"""
Application Service Provider
Laravel-style service provider for dependency injection
"""

from app.repositories.model_repository import ModelRepository
from app.services.prediction_service import PredictionService
from app.services.model_service import ModelService

class AppServiceProvider:
    """Service provider for application services"""
    
    def __init__(self, model_path: str = 'best_model.pkl'):
        self.model_path = model_path
        self._services = {}
        self._register_services()
    
    def _register_services(self):
        """Register all services"""
        # Register repository
        model_repository = ModelRepository(self.model_path)
        self._services['model_repository'] = model_repository
        
        # Register services
        self._services['prediction_service'] = PredictionService(model_repository)
        self._services['model_service'] = ModelService(model_repository)
    
    def get(self, service_name: str):
        """
        Get a service by name
        
        Args:
            service_name: Name of the service to retrieve
            
        Returns:
            Service instance
            
        Raises:
            KeyError: If service is not found
        """
        if service_name not in self._services:
            raise KeyError(f"Service '{service_name}' not found")
        return self._services[service_name]
    
    def load_model(self) -> bool:
        """
        Load the model through the repository
        
        Returns:
            bool: True if model loaded successfully
        """
        repository = self.get('model_repository')
        return repository.load()
    
    def get_prediction_service(self) -> PredictionService:
        """Get prediction service"""
        return self.get('prediction_service')
    
    def get_model_service(self) -> ModelService:
        """Get model service"""
        return self.get('model_service')
    
    def get_model_repository(self) -> ModelRepository:
        """Get model repository"""
        return self.get('model_repository')
