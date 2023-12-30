# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1_l2

# Load the CSV file
# https://finance.yahoo.com/quote/%5ERUT/history?period1=567734400&period2=1703894400&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true
file_path = 'data/^RUT.csv'  # Replace with your file path
stock_data = pd.read_csv(file_path)

# Selecting the 'Close' column
close_prices = stock_data['Close'].values.reshape(-1, 1)

# Normalizing the 'Close' prices
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_close = scaler.fit_transform(close_prices)


# Function to create a dataset with windows
def create_dataset(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size - 1):
        X.append(data[i:(i + window_size), 0])
        y.append(data[i + window_size, 0])
    return np.array(X), np.array(y)

# Defining a window size
window_size = 9000

# Create the dataset with windows
X, y = create_dataset(scaled_close, window_size)

# Reshaping input to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Building an advanced LSTM model
model = Sequential()
model.add(Bidirectional(LSTM(units=100, return_sequences=True, input_shape=(X.shape[1], 1), kernel_regularizer=l1_l2(l1=0.01, l2=0.01))))
model.add(Dropout(0.4))
model.add(Bidirectional(LSTM(units=100, return_sequences=False)))
model.add(Dropout(0.4))
model.add(Dense(units=1))

# Compiling the model with a different optimizer
model.compile(optimizer='rmsprop', loss='mean_squared_error')

# Implementing early stopping and potentially other callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Training the model with more epochs and potentially smaller batch size
history = model.fit(X, y, epochs=50, batch_size=16, verbose=1, validation_split=0.2, callbacks=[early_stopping])

# Summary of the model
model.summary()

# Predicting and making a trading decision
# Extract the last 9000 days of closing prices to make the next day's prediction
latest_data = stock_data['Close'].iloc[-9000:].values.reshape(-1, 1)

# Scale the latest data
latest_data_scaled = scaler.transform(latest_data)

# Reshape and predict
latest_data_scaled = latest_data_scaled.reshape((1, window_size, 1))
predicted_price_scaled = model.predict(latest_data_scaled)

# Inverse transform to get the actual predicted price
predicted_price = scaler.inverse_transform(predicted_price_scaled)

# Basic Trading Strategy
current_price = stock_data['Close'].iloc[-1]
predicted_price = predicted_price[0,0]

if predicted_price > current_price:
    print("Buy Signal - Predicted price is higher than the current price.")
elif predicted_price < current_price:
    print("Sell Signal - Predicted price is lower than the current price.")
else:
    print("Hold - No significant change in the predicted price.")