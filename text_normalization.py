"""
Text Normalization Module
Performs data cleaning, tokenization, and normalization for text classification
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class TextNormalizer:
    """Class for text normalization operations"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """
        Clean text by removing special characters, URLs, numbers, etc.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{10,}\b', '', text)
        
        # Remove special characters but keep spaces and apostrophes
        text = re.sub(r'[^a-zA-Z\s\']', ' ', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def tokenize(self, text):
        """
        Tokenize text into words
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        # Tokenize using NLTK
        tokens = word_tokenize(text)
        
        return tokens
    
    def normalize(self, tokens):
        """
        Normalize tokens by removing stopwords and applying stemming
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of normalized tokens
        """
        if not tokens:
            return []
        
        # Remove stopwords and apply stemming
        normalized_tokens = [
            self.stemmer.stem(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return normalized_tokens
    
    def process_text(self, text):
        """
        Complete text normalization pipeline
        
        Args:
            text: Input text string
            
        Returns:
            Normalized text string (joined tokens)
        """
        # Step 1: Clean text
        cleaned = self.clean_text(text)
        
        # Step 2: Tokenize
        tokens = self.tokenize(cleaned)
        
        # Step 3: Normalize
        normalized_tokens = self.normalize(tokens)
        
        # Join tokens back into string
        normalized_text = ' '.join(normalized_tokens)
        
        return normalized_text
    
    def process_dataframe(self, df, text_column='v2'):
        """
        Process entire dataframe column
        
        Args:
            df: Input dataframe
            text_column: Name of the text column
            
        Returns:
            Dataframe with normalized text column
        """
        df = df.copy()
        df['normalized_text'] = df[text_column].apply(self.process_text)
        return df

