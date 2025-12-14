"""
Test script for the Flask API
Demonstrates how to use the API endpoints
"""

import requests
import json

BASE_URL = "http://localhost:5000"

def test_home():
    """Test home endpoint"""
    print("=" * 60)
    print("Testing Home Endpoint")
    print("=" * 60)
    response = requests.get(f"{BASE_URL}/")
    print(json.dumps(response.json(), indent=2))
    print()

def test_health():
    """Test health endpoint"""
    print("=" * 60)
    print("Testing Health Endpoint")
    print("=" * 60)
    response = requests.get(f"{BASE_URL}/health")
    print(json.dumps(response.json(), indent=2))
    print()

def test_predict():
    """Test prediction endpoint"""
    print("=" * 60)
    print("Testing Prediction Endpoint")
    print("=" * 60)
    
    test_cases = [
        "Free entry in 2 a wkly comp to win FA Cup final tkts",
        "Hey, how are you doing today?",
        "URGENT! You have won a 1 week FREE membership",
        "I'll be home soon, see you later!",
        "The movie is not good"
    ]
    
    for text in test_cases:
        print(f"\nInput: {text}")
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"text": text}
        )
        result = response.json()
        print(f"Prediction: {result['prediction']}")
        print(f"Probabilities: Ham={result['probabilities']['ham']:.4f}, Spam={result['probabilities']['spam']:.4f}")
        print("-" * 60)

def test_batch_predict():
    """Test batch prediction endpoint"""
    print("=" * 60)
    print("Testing Batch Prediction Endpoint")
    print("=" * 60)
    
    texts = [
        "Free entry in 2 a wkly comp to win FA Cup final tkts",
        "Hey, how are you doing today?",
        "URGENT! You have won a 1 week FREE membership"
    ]
    
    response = requests.post(
        f"{BASE_URL}/predict_batch",
        json={"texts": texts}
    )
    result = response.json()
    print(f"Total predictions: {result['count']}")
    print("\nResults:")
    for i, res in enumerate(result['results'], 1):
        print(f"\n{i}. Input: {res['input_text']}")
        print(f"   Prediction: {res['prediction']}")
        print(f"   Probabilities: Ham={res['probabilities']['ham']:.4f}, Spam={res['probabilities']['spam']:.4f}")

if __name__ == "__main__":
    try:
        test_home()
        test_health()
        test_predict()
        test_batch_predict()
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to the API.")
        print("Please make sure the Flask server is running:")
        print("  python app.py")
    except Exception as e:
        print(f"ERROR: {str(e)}")

