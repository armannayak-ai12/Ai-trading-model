import requests
import pandas as pd
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------
# 🔑 API CONFIGURATION
# ---------------------------------------------------------
API_KEY = "d6auivhr01qnr27iv8igd6auivhr01qnr27iv8j0"  # <-- PASTE YOUR KEY HERE
# ---------------------------------------------------------

analyzer = SentimentIntensityAnalyzer()

def get_finnhub_news(symbol):
    """
    Fetches the last 3 days of news from Finnhub and calculates sentiment.
    """
    # Calculate dates (Last 3 days)
    today = datetime.now().strftime('%Y-%m-%d')
    three_days_ago = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    
    url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={three_days_ago}&to={today}&token={API_KEY}"
    
    try:
        response = requests.get(url)
        news_data = response.json()
        
        if not news_data:
            return pd.DataFrame(), 0
        
        processed_news = []
        total_score = 0
        count = 0
        
        # Analyze the top 10 newest articles
        for article in news_data[:10]:
            headline = article.get('headline', '')
            summary = article.get('summary', '')
            source = article.get('source', '')
            url = article.get('url', '')
            dt = datetime.fromtimestamp(article.get('datetime', 0)).strftime('%Y-%m-%d %H:%M')
            
            # Combine headline + summary for better accuracy
            full_text = f"{headline}. {summary}"
            sentiment = analyzer.polarity_scores(full_text)['compound']
            
            total_score += sentiment
            count += 1
            
            # Label
            if sentiment > 0.1:
                label = "Bullish 🟢"
            elif sentiment < -0.1:
                label = "Bearish 🔴"
            else:
                label = "Neutral ⚪"
            
            processed_news.append({
                "Date": dt,
                "Source": source,
                "Headline": headline,
                "Sentiment": label,
                "Score": sentiment,
                "Link": url
            })
            
        avg_score = total_score / count if count > 0 else 0
        
        return pd.DataFrame(processed_news), avg_score

    except Exception as e:
        print(f"Error fetching Finnhub news: {e}")
        return pd.DataFrame(), 0