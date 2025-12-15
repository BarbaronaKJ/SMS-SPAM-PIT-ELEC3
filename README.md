# SMS Spam Classification System

A complete text classification system that performs text normalization and classification using multiple machine learning models, with a Flask API for integration.

## Dataset

- **Dataset**: SMS Spam Collection Dataset
- **Rows**: 5,572 messages
- **Labels**: Spam (1) and Ham (0)
- **Location**: `dataset/spam.csv`

## Features

### 1. Text Normalization
- **Data Cleaning**: Removes URLs, email addresses, phone numbers, special characters
- **Tokenization**: Splits text into individual words using NLTK
- **Normalization**: Removes stopwords and applies stemming

### 2. Text Classification Models
The system trains and evaluates 5 different models:
1. **Naive Bayes** (MultinomialNB)
2. **Logistic Regression**
3. **Support Vector Machine (SVM)**
4. **Random Forest**
5. **K-Nearest Neighbors (KNN)**

The best model (based on F1-score) is automatically selected and saved.

### 3. Flask API with Web GUI
- **Web Interface**: Beautiful, modern web GUI accessible at `http://localhost:5000`
- **RESTful API endpoints**:
  - `GET /` - Web GUI interface
  - `GET /api` - API information
  - `GET /health` - Health check
  - `POST /predict` - Single text prediction
  - `POST /predict_batch` - Batch predictions

## Installation

1. Create and activate a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

**Note**: On Arch Linux and other systems with externally-managed Python environments, using a virtual environment is required.

2. Download NLTK data (automatically handled, but can be done manually):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Usage

### Step 1: Activate Virtual Environment (if not already activated)

```bash
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate
```

Or use the quick activation script:
```bash
source activate.sh
```

### Step 2: Train the Models

Run the classification script to train all models and select the best one:

```bash
python text_classification.py
```

This will:
- Load and normalize the dataset
- Train 5 different models
- Evaluate each model's performance
- Save the best model to `best_model.pkl`

### Step 3: Start the Flask API

```bash
python app.py
```

The API will start on `http://localhost:5000`

**🌐 Web GUI**: Open your browser and navigate to `http://localhost:5000` to use the interactive web interface!

**Quick Start**: You can also use the automated script:
```bash
./quick_start.sh
```

### Step 3: Use the API

#### Single Prediction

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Free entry in 2 a wkly comp to win FA Cup final tkts"}'
```

Response:
```json
{
  "input_text": "Free entry in 2 a wkly comp to win FA Cup final tkts",
  "normalized_text": "free entri wkli comp win fa cup final tkt",
  "prediction": "Spam",
  "probabilities": {
    "ham": 0.0234,
    "spam": 0.9766
  },
  "model_used": "Naive Bayes"
}
```

#### Batch Predictions

```bash
curl -X POST http://localhost:5000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Free entry in 2 a wkly comp", "Hey, how are you?"]}'
```

#### Health Check

```bash
curl http://localhost:5000/health
```

## Web GUI Features

The web interface provides:
- ✨ Modern, responsive design
- 📊 Real-time classification with probability visualization
- 💡 Example messages to try
- 📱 Mobile-friendly interface
- ⚡ Fast and intuitive user experience

Simply open `http://localhost:5000` in your browser after starting the Flask server!

## Example: Python Client

```python
import requests

# Single prediction
response = requests.post('http://localhost:5000/predict', 
                        json={'text': 'The movie is not good'})
result = response.json()
print(f"Prediction: {result['prediction']}")

# Batch predictions
response = requests.post('http://localhost:5000/predict_batch',
                        json={'texts': ['Text 1', 'Text 2']})
results = response.json()
```

## Example: JavaScript/Fetch

```javascript
fetch('http://localhost:5000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    text: 'The movie is not good'
  })
})
.then(response => response.json())
.then(data => {
  console.log('Prediction:', data.prediction);
  console.log('Probabilities:', data.probabilities);
});
```

## Project Structure

```
SMS-SPAM-PIT-elect3/
├── dataset/
│   └── spam.csv              # SMS spam dataset
├── templates/
│   └── index.html            # Web GUI interface
├── text_normalization.py     # Text normalization module
├── text_classification.py     # Model training and evaluation
├── app.py                    # Flask API server with web GUI
├── best_model.pkl            # Saved best model (generated after training)
├── requirements.txt          # Python dependencies
├── test_api.py               # API test script
├── test_client.html          # Standalone test client
└── README.md                 # This file
```

## Model Performance

After training, you'll see detailed performance metrics for each model including:
- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report
- Confusion Matrix

The best model is automatically selected based on F1-score.

## API Endpoints

### GET /
Serves the web GUI interface. Open in your browser for interactive classification.

### GET /api
Returns API information and available endpoints in JSON format.

### GET /health
Returns API health status and model loading status.

### POST /predict
**Request Body:**
```json
{
  "text": "Your text message here"
}
```

**Response:**
```json
{
  "input_text": "Your text message here",
  "normalized_text": "normalized version",
  "prediction": "Spam" or "Ham",
  "probabilities": {
    "ham": 0.95,
    "spam": 0.05
  },
  "model_used": "Model Name"
}
```

### POST /predict_batch
**Request Body:**
```json
{
  "texts": ["Text 1", "Text 2", "Text 3"]
}
```

**Response:**
```json
{
  "results": [
    {
      "input_text": "Text 1",
      "normalized_text": "...",
      "prediction": "Spam",
      "probabilities": {...}
    },
    ...
  ],
  "count": 3,
  "model_used": "Model Name"
}
```

## Notes

- The dataset is automatically normalized before training
- The model uses TF-IDF vectorization with 5000 features
- Text normalization includes cleaning, tokenization, stopword removal, and stemming
- CORS is enabled for cross-origin requests
- The API runs on port 5000 by default

## License

This project is for educational purposes.

