import numpy as np
import matplotlib.pyplot as plt

# --- 1. Simulation Parameters ---
np.random.seed(42)
T = 1.0
N = 252
dt = T / N
S0 = 100.0

# Standard Diffusion Parameters
mu = 0.05
sigma = 0.20

# Modern Jump Parameters (The Risk Shocks)
lambda_jumps = 3.0    # Expected number of catastrophic jumps per year
mu_jumps = -0.15     # Average jump size (e.g., a sudden -15% drop)
sigma_jumps = 0.10  # Volatility of the jump size

# --- 2. Simulating the Process ---
num_paths = 5
S = np.zeros((N + 1, num_paths))
S[0] = S0

for path in range(num_paths):
    # Standard Brownian Randomness
    Z = np.random.standard_normal(N)
    
    # Jump Randomness (Poisson Process)
    # Determines if a jump happens at each time step
    num_jumps_in_step = np.random.poisson(lambda_jumps * dt, N)
    
    for t in range(1, N + 1):
        # Calculate standard diffusion drift and noise
        diffusion_part = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t-1]
        
        # Calculate Jump impact if a Poisson event is triggered
        jump_part = 0
        if num_jumps_in_step[t-1] > 0:
            # Generate a random negative shock size
            jump_part = np.sum(np.random.normal(mu_jumps, sigma_jumps, num_jumps_in_step[t-1]))
        
        # Combine standard diffusion with sudden jumps
        S[t, path] = S[t-1, path] * np.exp(diffusion_part + jump_part)

# --- 3. Plotting the Modern Risk Simulation ---
plt.figure(figsize=(10, 6))
plt.plot(S)
plt.title("Toy Risk Simulation: Jump-Diffusion Model")
plt.xlabel("Trading Days")
plt.ylabel("Price ($)")
plt.grid(True)
plt.show()
