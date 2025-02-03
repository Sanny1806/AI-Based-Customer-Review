# AI-Based-Customer-Review
Sentiment Analysis API

A simple Flask API for sentiment analysis using VADER and RoBERTa.
Used Amazon Reviews dataset from Kaggle.

Features:
- Uses VADER for rule-based sentiment analysis.
- Uses RoBERTa for deep-learning sentiment classification.

Installation:
- Install dependencies: pip install -r requirements.txt
- Run the app: python app.py

Usage:
- GET "/" → Welcome message
- POST "/sentiment" → Send JSON { "text": "your text here" }  
  Returns sentiment scores from both models.

Models:
- VADER (Lexicon-based)
- RoBERTa (Pre-trained on Twitter data)

Open-source project. Modify and use freely.
