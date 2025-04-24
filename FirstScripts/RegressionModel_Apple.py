import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import yfinance as yf
import matplotlib.pyplot as plt

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

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, predictions, label='Predicted')
plt.title(f"Linear Regression Prediction for {ticker} Stock Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

