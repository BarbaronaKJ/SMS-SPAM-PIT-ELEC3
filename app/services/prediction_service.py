"""
Prediction Service
Business logic for text classification predictions
"""

from typing import Dict, List, Any, Optional
from app.repositories.model_repository import ModelRepository

class PredictionService:
    """Service for handling prediction operations"""
    
    def __init__(self, model_repository: ModelRepository):
        self.model_repository = model_repository
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict if text is spam or ham
        
        Args:
            text: Input text to classify
            
        Returns:
            Dict containing prediction results
            
        Raises:
            ValueError: If model is not loaded or text is invalid
        """
        if not self.model_repository.is_loaded():
            raise ValueError('Model not loaded. Please train the model first.')
        
        if not text or not isinstance(text, str):
            raise ValueError('Text must be a non-empty string')
        
        # Normalize text
        normalizer = self.model_repository.get_normalizer()
        normalized_text = normalizer.process_text(text)
        
        if not normalized_text:
            raise ValueError('Text could not be processed. Please provide valid text.')
        
        # Vectorize
        vectorizer = self.model_repository.get_vectorizer()
        text_vectorized = vectorizer.transform([normalized_text])
        
        # Predict
        model = self.model_repository.get_model()
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]
        
        # Format result
        return {
            'input_text': text,
            'normalized_text': normalized_text,
            'prediction': 'Spam' if prediction == 1 else 'Ham',
            'probabilities': {
                'ham': float(probabilities[0]),
                'spam': float(probabilities[1])
            },
            'model_used': self.model_repository.get_model_name()
        }
    
    def predict_batch(self, texts: List[str]) -> Dict[str, Any]:
        """
        Predict for multiple texts
        
        Args:
            texts: List of texts to classify
            
        Returns:
            Dict containing batch prediction results
            
        Raises:
            ValueError: If model is not loaded or texts are invalid
        """
        if not self.model_repository.is_loaded():
            raise ValueError('Model not loaded. Please train the model first.')
        
        if not isinstance(texts, list) or len(texts) == 0:
            raise ValueError('Texts must be a non-empty array')
        
        results = []
        
        for text in texts:
            if not isinstance(text, str):
                continue
            
            try:
                result = self.predict(text)
                results.append(result)
            except ValueError:
                # Skip invalid texts
                continue
        
        return {
            'results': results,
            'count': len(results),
            'model_used': self.model_repository.get_model_name()
        }
