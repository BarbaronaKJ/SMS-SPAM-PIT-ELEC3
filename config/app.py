"""
Application Configuration
Laravel-style configuration management
"""

import os

class Config:
    """Base configuration"""
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    PORT = int(os.getenv('FLASK_PORT', 5000))
    
    # Model configuration
    MODEL_PATH = os.getenv('MODEL_PATH', 'best_model.pkl')
    
    # CORS configuration
    CORS_ENABLED = True
    
    # API configuration
    API_PREFIX = '/api'
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config(env=None):
    """Get configuration based on environment"""
    env = env or os.getenv('FLASK_ENV', 'default')
    return config.get(env, config['default'])
