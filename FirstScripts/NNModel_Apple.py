import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf

# Download historical stock data for a single company (e.g., Apple)
ticker = "AAPL"
data = yf.download(ticker, start="2020-01-01", end="2024-01-01")

# Use main financial indicators for modeling
features = ['Open', 'High', 'Low', 'Volume']
target = 'Close'

# Drop missing values
stock_data = data[features + [target]].dropna()

# Split data into training and test sets
X = stock_data[features]
y = stock_data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a simple Neural Network with TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # No activation for regression output
])

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=1, validation_split=0.1)

# Predict and evaluate
predictions = model.predict(X_test_scaled).flatten()
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error (Neural Network): {mse:.2f}")

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, predictions, label='Predicted')
plt.title(f"Neural Network Prediction for {ticker} Stock Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()
