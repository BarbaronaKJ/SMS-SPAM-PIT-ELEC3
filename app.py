"""
Flask API for SMS Spam Classification
Laravel-style architecture with service providers, services, and repositories
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from config.app import get_config
from app.providers.app_service_provider import AppServiceProvider
from app.middleware.error_handler import handle_errors

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Load configuration
config = get_config()
app.config.from_object(config)

# Enable CORS
if config.CORS_ENABLED:
    CORS(app)

# Initialize service provider (Laravel-style dependency injection)
service_provider = AppServiceProvider(model_path=config.MODEL_PATH)

@app.route('/', methods=['GET'])
def home():
    """Home endpoint - serves web GUI"""
    return render_template('index.html')

@app.route('/api', methods=['GET'])
def api_info():
    """API information endpoint"""
    model_service = service_provider.get_model_service()
    return jsonify(model_service.get_api_info())

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    model_service = service_provider.get_model_service()
    return jsonify(model_service.get_health_status())

@app.route('/predict', methods=['POST'])
@handle_errors
def predict():
    """
    Predict endpoint
    Accepts JSON with 'text' field
    Returns prediction (Spam or Ham)
    """
    # Get JSON data
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({
            'error': 'Please provide "text" field in JSON body'
        }), 400
    
    text = data['text']
    
    # Use prediction service (Laravel-style service layer)
    prediction_service = service_provider.get_prediction_service()
    result = prediction_service.predict(text)
    
    return jsonify(result)

@app.route('/predict_batch', methods=['POST'])
@handle_errors
def predict_batch():
    """
    Batch prediction endpoint
    Accepts JSON with 'texts' field (array of strings)
    Returns predictions for all texts
    """
    # Get JSON data
    data = request.get_json()
    
    if not data or 'texts' not in data:
        return jsonify({
            'error': 'Please provide "texts" field (array) in JSON body'
        }), 400
    
    texts = data['texts']
    
    # Use prediction service (Laravel-style service layer)
    prediction_service = service_provider.get_prediction_service()
    result = prediction_service.predict_batch(texts)
    
    return jsonify(result)

if __name__ == '__main__':
    print("Loading model...")
    if service_provider.load_model():
        model_repository = service_provider.get_model_repository()
        model_name = model_repository.get_model_name()
        print(f"Model loaded successfully: {model_name}")
        print("\nAPI Endpoints:")
        print("  GET  /          - Web GUI")
        print("  GET  /api       - API information")
        print("  GET  /health    - Health check")
        print("  POST /predict   - Single prediction")
        print("  POST /predict_batch - Batch predictions")
        print("\nStarting Flask server...")
        print(f"\n🌐 Web GUI available at: http://{config.HOST}:{config.PORT}")
        print(f"📡 API available at: http://{config.HOST}:{config.PORT}/api")
        app.run(debug=config.DEBUG, host=config.HOST, port=config.PORT)
    else:
        print(f"ERROR: Model file '{config.MODEL_PATH}' not found!")
        print("Please run 'python text_classification.py' first to train the model.")

