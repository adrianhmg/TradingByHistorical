import praw
from textblob import TextBlob
import nltk

class SentimentAnalyzer:
    def __init__(self, client_id, client_secret, user_agent):
        self.reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
        nltk.download('punkt')

    def get_reddit_posts(self, subreddit, query, limit=100):
        posts = self.reddit.subreddit(subreddit).search(query, limit=limit)
        return [post.title for post in posts]

    def analyze_sentiment(self, posts):
        sentiments = []
        for post in posts:
            analysis = TextBlob(post)
            sentiments.append(analysis.sentiment.polarity)
        return sum(sentiments) / len(sentiments) if sentiments else 0