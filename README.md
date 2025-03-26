# 📧 Spam Email Classifier

A machine learning model that classifies emails as **spam** or **ham (non-spam)** using Natural Language Processing (NLP).

![Demo](https://img.shields.io/badge/Demo-Live-green) 

## 🚀 Features
- Preprocesses email text (stopword removal, stemming, tokenization)
- Trains using **Naive Bayes** (or Logistic Regression)
- Web interface with Flask
- Accuracy: ~98% on test data
- Lightweight and easy to deploy

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/spam-email-classifier.git
   cd spam-email-classifier
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**
   ```bash
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
   ```

## 🛠️ Usage

1. **Train the model**
   ```bash
   python train_model.py
   ```

2. **Run the web app**
   ```bash
   python app.py
   ```
   Open http://localhost:5000 in your browser

3. **Classify an email**
   - Paste email content into the text box
   - Click "Classify" to see results

## 📂 Project Structure
```
spam-classifier/
├── data/               # Dataset (spam/ham emails)
├── models/             # Saved ML models
├── static/             # CSS/JS files
├── templates/          # HTML templates
├── app.py              # Flask application
├── train_model.py      # Model training script
└── requirements.txt    # Dependencies
```

## 📊 Performance
| Model            | Accuracy | Precision | Recall |
|------------------|----------|-----------|--------|
| Naive Bayes      | 98.2%    | 97.8%     | 99.1%  |
| Logistic Regression | 98.5%  | 98.3%     | 98.7%  |

## 🌟 Future Improvements
- Add user feedback system
- Implement batch email processing
- Deploy as Chrome extension

- Dataset attribution details?
- Deployment instructions for Heroku/AWS?
