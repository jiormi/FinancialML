import numpy as np
import matplotlib.pyplot as plt

# --- 1. Simulation Parameters ---
S0 = 100.0       # Initial stock price
mu = 0.05        # Drift (expected annual return, e.g., 5%)
sigma = 0.20     # Volatility (annual standard deviation, e.g., 20%)
T = 1.0          # Time horizon in years (e.g., 1 year)
N = 252          # Number of trading days/steps
dt = T / N       # Time step size
num_paths = 5    # Number of simulated price paths to generate

# --- 2. Generate Randomness ---
# Seed for reproducibility
np.random.seed(42) 

# Generate independent standard normal random variables for each step and path
# Standard normal distribution represents the dW_t term
Z = np.random.standard_normal((N, num_paths))

# --- 3. Compute the Diffusion Process ---
# Create an array to store the price paths, starting with S0
S = np.zeros((N + 1, num_paths))
S[0] = S0

# Apply the Geometric Brownian Motion formula step-by-step
for t in range(1, N + 1):
    # S_t = S_{t-1} * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)
    S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t-1])

# --- 4. Plot the Results ---
plt.figure(figsize=(10, 6))
plt.plot(S)
plt.title(f"Geometric Brownian Motion: {num_paths} Simulated Stock Price Paths")
plt.xlabel("Trading Days")
plt.ylabel("Stock Price ($)")
plt.grid(True)
plt.show()
