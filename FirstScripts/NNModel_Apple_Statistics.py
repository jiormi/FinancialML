import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf

# Parameters
ticker = "AAPL"
api_key = "OtkFx4zvN7fJLbFKcxpVHZv2huNp7bTF"  # Replace with your FMP API key

# Download historical stock data for a single company (e.g., Apple)
data = yf.download(ticker, start="2020-01-01", end="2024-01-01")

# Add rolling features for trend analysis
data['MA5'] = data['Close'].rolling(window=5).mean()
data['MA10'] = data['Close'].rolling(window=10).mean()
data['MA20'] = data['Close'].rolling(window=20).mean()

# Drop rows with NaNs after rolling calculations
data.dropna(inplace=True)

# Fetch financial statement growth data from FMP
def fetch_growth_data(symbol, api_key):
    url = f"https://financialmodelingprep.com/api/v3/financial-growth/{symbol}?apikey={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to fetch growth data")
    growth_data = response.json()[0]
    return {
        'revenueGrowth': growth_data.get('revenueGrowth'),
        'netIncomeGrowth': growth_data.get('netIncomeGrowth'),
        'epsgrowth': growth_data.get('epsgrowth'),
        'dividendsPerShareGrowth': growth_data.get('dividendsPerShareGrowth'),
        'freeCashFlowGrowth': growth_data.get('freeCashFlowGrowth'),
        'assetGrowth': growth_data.get('assetGrowth'),
        'bookValueperShareGrowth': growth_data.get('bookValueperShareGrowth'),
        'debtGrowth': growth_data.get('debtGrowth'),
        'operatingCashFlowGrowth': growth_data.get('operatingCashFlowGrowth'),
        'rdexpenseGrowth': growth_data.get('rdexpenseGrowth')
    }

# Insert real growth fundamentals as static features for now
growth_features = fetch_growth_data(ticker, api_key)
for key, value in growth_features.items():
    data[key] = value

# Define features and target
features = ['Open', 'High', 'Low', 'Close', 'Volume',
            'MA5', 'MA10', 'MA20'] + list(growth_features.keys())
target = 'Close'

# Prepare data
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a simple Neural Network with TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1)  # No activation for regression output
])

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=1, validation_split=0.1)

# Predict and evaluate
predictions = model.predict(X_test_scaled).flatten()
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"Mean Squared Error (Neural Network): {mse:.2f}")
print(f"Mean Absolute Error (Neural Network): {mae:.2f}")
print(f"R-squared Score (Explained Variance): {r2:.4f}")

# Bootstrap confidence interval for MSE
boot_mses = []
for _ in range(100):
    idxs = np.random.choice(len(X_test_scaled), size=len(X_test_scaled), replace=True)
    boot_preds = model.predict(X_test_scaled[idxs]).flatten()
    boot_mses.append(mean_squared_error(y_test.values[idxs], boot_preds))

ci_lower = np.percentile(boot_mses, 2.5)
ci_upper = np.percentile(boot_mses, 97.5)
print(f"95% Confidence Interval for MSE: [{ci_lower:.2f}, {ci_upper:.2f}]")

# Relative Error (normalized by variance of target)
baseline_mse = np.var(y_test.values)
relative_error = mse / baseline_mse
print(f"Relative Error (MSE / Variance): {relative_error:.4f}")

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, predictions, label='Predicted')
plt.title(f"Neural Network Prediction for {ticker} Stock Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# Residual plot
residuals = y_test.values - predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test.index, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residual Plot: Prediction Errors Over Time")
plt.xlabel("Date")
plt.ylabel("Residual (Actual - Predicted)")
plt.grid(True)
plt.show()

# --- Statistical Foundation Note ---
# Neural networks are non-linear, non-parametric models. Traditional statistical tests
# (like t-tests or R²) assume linearity and normally distributed residuals, which do not hold here.
# Hence, statistical interpretation is better approached through empirical methods:
# - Bootstrapping: estimate variability/confidence intervals of the metric
# - Relative error: compare model performance to baseline (e.g. predicting the mean)
# - Dropout-based uncertainty: treat dropout as Bayesian inference at test time (MC Dropout)
# These methods help quantify prediction reliability without strict distributional assumptions.
# - MAE is more robust to outliers than MSE.
# - Residual plots can help reveal structure not captured by the model.

