import numpy as np
import matplotlib.pyplot as plt

# --- 1. Simulation Setup ---
np.random.seed(10)
T = 1.0          # Time horizon (1 year)
N = 1000         # High resolution to approximate continuous integration
dt = T / N       # Time step
time = np.linspace(0, T, N + 1)

# Stock Parameters
S0 = 100.0
mu = 0.10        # 10% expected annual drift
sigma = 0.30     # 30% volatility

# --- 2. Simulate the Stock Price (Diffusion Process) ---
S = np.zeros(N + 1)
S[0] = S0
Z = np.random.standard_normal(N)

for t in range(1, N + 1):
    S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t-1])

# --- 3. The Trading Strategy & Itô Integration ---
# Strategy: Hold shares proportional to the stock price change.
# H[t] represents the number of shares held *before* the price moves to S[t+1]
H = np.zeros(N) 
PnL_accumulated = np.zeros(N + 1) # Total profit/loss tracking

for t in range(0, N):
    # Decision Rule (Looking Backward): Look at current price to decide shares
    # If price is above $100, buy 1 share for every dollar above. If below, short it.
    H[t] = 2.0 * (S[t] - 100.0) 
    
    # The Ito Integral Step: Profit = Shares * Change in Price
    # dS = S[t+1] - S[t]. This contains the random shock!
    dS = S[t+1] - S[t]
    
    # Accumulate the profit over time (Integrating H_t * dS_t)
    PnL_accumulated[t+1] = PnL_accumulated[t] + (H[t] * dS)

# --- 4. Plotting the Simulation ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot 1: Stock Price
ax1.plot(time, S, color='blue', label='Stock Price ($S_t$)')
ax1.axhline(100, color='gray', linestyle='--', alpha=0.7, label='Baseline ($100)')
ax1.set_title("The Underlying Diffusion Process (Stock Price)")
ax1.set_ylabel("Price ($)")
ax1.legend(loc='upper left')
ax1.grid(True)

# Plot 2: Cumulative Profit (The Integral Result)
ax2.plot(time, PnL_accumulated, color='green', label='Cumulative Portfolio PnL')
ax2.set_title("The Result of the Itô Integral: $\int H_t \, dS_t$ (Trading Profit/Loss)")
ax2.set_xlabel("Time (Years)")
ax2.set_ylabel("Total Profit/Loss ($)")
ax2.legend(loc='upper left')
ax2.grid(True)

plt.tight_layout()
plt.show()
