from flask import Flask, render_template, request, jsonify
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import os

app = Flask(__name__)

# Load the model
model_path = os.path.join('models', 'spam_classifier.pkl')
model = joblib.load(model_path)

# Ensure NLTK data is available
nltk.download('stopwords')
nltk.download('punkt')

# Text preprocessing (same as during training)
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email_text = request.form['email_text']
        processed_text = preprocess_text(email_text)
        prediction = model.predict([processed_text])[0]
        probability = model.predict_proba([processed_text])[0].max()
        
        result = {
            'prediction': 'spam' if prediction == 1 else 'ham',
            'probability': round(probability * 100, 2)
        }
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)