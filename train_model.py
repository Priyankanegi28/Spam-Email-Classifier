import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import os

# Ensure NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Load dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    # Handle different column names in different datasets
    if 'label' in df.columns and 'text' in df.columns:
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    elif 'Category' in df.columns and 'Message' in df.columns:
        df['label'] = df['Category'].map({'ham': 0, 'spam': 1})
        df['text'] = df['Message']
    else:
        raise ValueError("Dataset format not recognized")
    return df[['text', 'label']]

# Text preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # Tokenize
    words = nltk.word_tokenize(text)
    
    # Remove stopwords and stem
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

# Train and save model
def train_and_save_model():
    # Paths
    raw_data_path = os.path.join('data', 'raw', 'spam_ham_dataset.csv')
    model_path = os.path.join('models', 'spam_classifier.pkl')
    
    # Load and preprocess data
    df = load_data(raw_data_path)
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['label'], test_size=0.2, random_state=42
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train_and_save_model()