"""
Text Classification Module
Implements multiple models for text classification with training, testing, and evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import pickle
import os
from text_normalization import TextNormalizer

class TextClassifier:
    """Class for text classification with multiple models"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.models = {}
        self.model_scores = {}
        self.best_model = None
        self.best_model_name = None
        self.normalizer = TextNormalizer()
        
    def prepare_data(self, df, text_column='normalized_text', label_column='v1'):
        """
        Prepare data for training
        
        Args:
            df: Input dataframe
            text_column: Name of the text column
            label_column: Name of the label column
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Encode labels (spam=1, ham=0)
        df['label_encoded'] = df[label_column].map({'spam': 1, 'ham': 0})
        
        # Split data
        X = df[text_column].values
        y = df['label_encoded'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Vectorize text
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        return X_train_vectorized, X_test_vectorized, y_train, y_test
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """
        Train multiple classification models
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training labels
            y_test: Testing labels
        """
        # Define models
        models_to_train = {
            'Naive Bayes': MultinomialNB(alpha=1.0),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'SVM': SVC(kernel='linear', random_state=42, probability=True),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
        }
        
        print("=" * 60)
        print("Training Multiple Classification Models")
        print("=" * 60)
        
        for name, model in models_to_train.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Store model and scores
            self.models[name] = model
            self.model_scores[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            # Print results
            print(f"{name} Results:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
            print(f"\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
            print("-" * 60)
        
        # Select best model based on F1-score
        self.best_model_name = max(self.model_scores, key=lambda x: self.model_scores[x]['f1_score'])
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n{'=' * 60}")
        print(f"Best Model: {self.best_model_name}")
        print(f"F1-Score: {self.model_scores[self.best_model_name]['f1_score']:.4f}")
        print(f"{'=' * 60}\n")
    
    def predict(self, text):
        """
        Predict label for a single text
        
        Args:
            text: Input text string
            
        Returns:
            Predicted label (0 for ham, 1 for spam)
        """
        # Normalize text
        normalized_text = self.normalizer.process_text(text)
        
        # Vectorize
        text_vectorized = self.vectorizer.transform([normalized_text])
        
        # Predict
        prediction = self.best_model.predict(text_vectorized)[0]
        
        return prediction
    
    def predict_proba(self, text):
        """
        Get prediction probabilities
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with probabilities
        """
        # Normalize text
        normalized_text = self.normalizer.process_text(text)
        
        # Vectorize
        text_vectorized = self.vectorizer.transform([normalized_text])
        
        # Get probabilities
        probabilities = self.best_model.predict_proba(text_vectorized)[0]
        
        return {
            'ham': probabilities[0],
            'spam': probabilities[1]
        }
    
    def save_model(self, filepath='best_model.pkl'):
        """
        Save the best model and vectorizer
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'model': self.best_model,
            'vectorizer': self.vectorizer,
            'normalizer': self.normalizer,
            'model_name': self.best_model_name,
            'scores': self.model_scores
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='best_model.pkl'):
        """
        Load a saved model
        
        Args:
            filepath: Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.best_model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.normalizer = model_data['normalizer']
        self.best_model_name = model_data['model_name']
        self.model_scores = model_data['scores']
        
        print(f"Model loaded from {filepath}")

def main():
    """Main function to run text classification"""
    
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('dataset/spam.csv', encoding='latin-1')
    
    # Keep only relevant columns
    df = df[['v1', 'v2']]
    df.columns = ['v1', 'v2']
    
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution:\n{df['v1'].value_counts()}\n")
    
    # Normalize text
    print("Performing text normalization...")
    normalizer = TextNormalizer()
    df = normalizer.process_dataframe(df, text_column='v2')
    
    # Remove empty normalized texts
    df = df[df['normalized_text'].str.len() > 0]
    
    print(f"Dataset shape after normalization: {df.shape}\n")
    
    # Initialize classifier
    classifier = TextClassifier()
    
    # Prepare data
    print("Preparing data for training...")
    X_train, X_test, y_train, y_test = classifier.prepare_data(df)
    
    # Train models
    classifier.train_models(X_train, X_test, y_train, y_test)
    
    # Save model
    classifier.save_model('best_model.pkl')
    
    # Test predictions
    print("\n" + "=" * 60)
    print("Testing Predictions on Sample Texts")
    print("=" * 60)
    
    test_texts = [
        "Free entry in 2 a wkly comp to win FA Cup final tkts",
        "Hey, how are you doing today?",
        "URGENT! You have won a 1 week FREE membership",
        "I'll be home soon, see you later!"
    ]
    
    for text in test_texts:
        prediction = classifier.predict(text)
        proba = classifier.predict_proba(text)
        label = "Spam" if prediction == 1 else "Ham"
        print(f"\nText: {text}")
        print(f"Prediction: {label}")
        print(f"Probabilities: Ham={proba['ham']:.4f}, Spam={proba['spam']:.4f}")

if __name__ == "__main__":
    main()

