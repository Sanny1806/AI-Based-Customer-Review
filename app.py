from flask import Flask, request, jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

app = Flask(__name__)

# Initialize the VADER SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Initialize the RoBERTa model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
model = RobertaForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')

@app.route('/')
def welcome():
    return "Welcome to sentiment analysis api"

@app.route('/sentiment', methods=['POST'])
def sentiment():
    data = request.get_json()
    text = data['text']
    
    # VADER Sentiment Analysis
    vader_score = analyzer.polarity_scores(text)['compound']
    
    # RoBERTa Sentiment Analysis
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    roberta_score = torch.softmax(logits, dim=-1).detach().numpy()
    roberta_sentiment = roberta_score.argmax()

    # Mapping RoBERTa sentiment to label
    if roberta_sentiment == 0:
        roberta_sentiment_label = "Negative"
    elif roberta_sentiment == 1:
        roberta_sentiment_label = "Neutral"
    else:
        roberta_sentiment_label = "Positive"

    return jsonify({
        'vader_score': vader_score,
        'roberta_sentiment': roberta_sentiment_label,
        'roberta_score': roberta_score.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)




