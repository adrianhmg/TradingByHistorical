import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1_l2

class StockPricePredictor:
    def __init__(self, file_path, window_size=9000):
        self.file_path = file_path
        self.window_size = window_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = self.build_model()

    def load_and_preprocess_data(self):
        stock_data = pd.read_csv(self.file_path)
        close_prices = stock_data['Close'].values.reshape(-1, 1)
        scaled_close = self.scaler.fit_transform(close_prices)
        return scaled_close

    def create_dataset(self, data):
        X, y = [], []
        for i in range(len(data) - self.window_size - 1):
            X.append(data[i:(i + self.window_size), 0])
            y.append(data[i + self.window_size, 0])
        X = np.array(X)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshaping to [samples, time steps, features]
        return X, np.array(y)

    def build_model(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(units=100, return_sequences=True, input_shape=(self.window_size, 1), kernel_regularizer=l1_l2(l1=0.01, l2=0.01))))
        model.add(Dropout(0.4))
        model.add(Bidirectional(LSTM(units=100, return_sequences=False)))
        model.add(Dropout(0.4))
        model.add(Dense(units=1))
        model.compile(optimizer='rmsprop', loss='mean_squared_error')
        return model

    def train_model(self, X, y):
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        self.model.fit(X, y, epochs=10, batch_size=16, verbose=1, validation_split=0.2, callbacks=[early_stopping])

    def predict(self, latest_data):
        latest_data_scaled = self.scaler.transform(latest_data.reshape(-1, 1))
        latest_data_scaled = latest_data_scaled.reshape((1, self.window_size, 1))
        predicted_price_scaled = self.model.predict(latest_data_scaled)
        predicted_price = self.scaler.inverse_transform(predicted_price_scaled)
        return predicted_price[0, 0]