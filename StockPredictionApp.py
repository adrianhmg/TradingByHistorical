from SentimentAnalyzer import SentimentAnalyzer
from StockPricePredictor import StockPricePredictor
from config import REDDIT_CREDENTIALS

class StockPredictionApp:
    def __init__(self, stock_file_path, reddit_credentials):
        self.stock_predictor = StockPricePredictor(stock_file_path)
        self.sentiment_analyzer = SentimentAnalyzer(**reddit_credentials)

    def run(self):
        # Load and preprocess stock data
        scaled_data = self.stock_predictor.load_and_preprocess_data()
        X, y = self.stock_predictor.create_dataset(scaled_data)

        # Train the stock price prediction model
        self.stock_predictor.train_model(X, y)

        # Fetch and analyze Reddit sentiments
        posts = self.sentiment_analyzer.get_reddit_posts('stocks', 'Russell 2000', 70)
        avg_sentiment = self.sentiment_analyzer.analyze_sentiment(posts)
        print(f"Average Sentiment Score: {avg_sentiment}")

        # Predict the next day's price
        latest_data = scaled_data[-self.stock_predictor.window_size:]
        predicted_price = self.stock_predictor.predict(latest_data)

        # Display the predicted price
        print(f"Predicted Price for the next day: {predicted_price}")

        # Make a decision based on the predicted price and sentiment
        current_price = scaled_data[-1, 0]
        if avg_sentiment > 0.1 and predicted_price > current_price:
            decision = "Strong Buy"
        elif avg_sentiment < -0.1 and predicted_price < current_price:
            decision = "Strong Sell"
        elif predicted_price > current_price:
            decision = "Buy"
        elif predicted_price < current_price:
            decision = "Sell"
        else:
            decision = "Hold"

        print(f"Recommended Action: {decision}")

# Application usage example
app = StockPredictionApp('data/^RUT.csv', REDDIT_CREDENTIALS)
app.run()