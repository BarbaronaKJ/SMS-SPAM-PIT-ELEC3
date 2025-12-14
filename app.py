"""
Flask API for SMS Spam Classification
Simple API endpoint to classify text messages as Spam or Ham
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import os
from text_normalization import TextNormalizer

app = Flask(__name__, template_folder='templates')
CORS(app)  # Enable CORS for cross-origin requests

# Global variables for model
model = None
vectorizer = None
normalizer = None
model_name = None

def load_model():
    """Load the trained model"""
    global model, vectorizer, normalizer, model_name
    
    if not os.path.exists('best_model.pkl'):
        return False
    
    with open('best_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    vectorizer = model_data['vectorizer']
    normalizer = model_data['normalizer']
    model_name = model_data.get('model_name', 'Unknown')
    
    return True

@app.route('/', methods=['GET'])
def home():
    """Home endpoint - serves web GUI"""
    return render_template('index.html')

@app.route('/api', methods=['GET'])
def api_info():
    """API information endpoint"""
    return jsonify({
        'message': 'SMS Spam Classification API',
        'status': 'running',
        'model': model_name if model_name else 'Not loaded',
        'endpoints': {
            '/predict': 'POST - Classify text (body: {"text": "your text here"})',
            '/health': 'GET - Check API health',
            '/api': 'GET - API information'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict endpoint
    Accepts JSON with 'text' field
    Returns prediction (Spam or Ham)
    """
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Please provide "text" field in JSON body'
            }), 400
        
        text = data['text']
        
        if not text or not isinstance(text, str):
            return jsonify({
                'error': 'Text must be a non-empty string'
            }), 400
        
        # Normalize text
        normalized_text = normalizer.process_text(text)
        
        if not normalized_text:
            return jsonify({
                'error': 'Text could not be processed. Please provide valid text.'
            }), 400
        
        # Vectorize
        text_vectorized = vectorizer.transform([normalized_text])
        
        # Predict
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]
        
        # Format result
        result = {
            'input_text': text,
            'normalized_text': normalized_text,
            'prediction': 'Spam' if prediction == 1 else 'Ham',
            'probabilities': {
                'ham': float(probabilities[0]),
                'spam': float(probabilities[1])
            },
            'model_used': model_name
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        }), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint
    Accepts JSON with 'texts' field (array of strings)
    Returns predictions for all texts
    """
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                'error': 'Please provide "texts" field (array) in JSON body'
            }), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({
                'error': 'Texts must be a non-empty array'
            }), 400
        
        results = []
        
        for text in texts:
            if not isinstance(text, str):
                continue
            
            # Normalize text
            normalized_text = normalizer.process_text(text)
            
            if not normalized_text:
                continue
            
            # Vectorize
            text_vectorized = vectorizer.transform([normalized_text])
            
            # Predict
            prediction = model.predict(text_vectorized)[0]
            probabilities = model.predict_proba(text_vectorized)[0]
            
            results.append({
                'input_text': text,
                'normalized_text': normalized_text,
                'prediction': 'Spam' if prediction == 1 else 'Ham',
                'probabilities': {
                    'ham': float(probabilities[0]),
                    'spam': float(probabilities[1])
                }
            })
        
        return jsonify({
            'results': results,
            'count': len(results),
            'model_used': model_name
        })
    
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("Loading model...")
    if load_model():
        print(f"Model loaded successfully: {model_name}")
        print("\nAPI Endpoints:")
        print("  GET  /          - Web GUI")
        print("  GET  /api       - API information")
        print("  GET  /health    - Health check")
        print("  POST /predict   - Single prediction")
        print("  POST /predict_batch - Batch predictions")
        print("\nStarting Flask server...")
        print(f"\n🌐 Web GUI available at: http://localhost:5000")
        print(f"📡 API available at: http://localhost:5000/api")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("ERROR: Model file 'best_model.pkl' not found!")
        print("Please run 'python text_classification.py' first to train the model.")

