import numpy as np
import matplotlib.pyplot as plt

# 1. Setup parameters
num_steps = 1000  # Number of steps the particle takes
step_size = 1.0   # How far it can move in a single step

# 2. Generate random steps for X and Y directions
# np.random.normal generates random numbers forming a bell curve (Gaussian distribution)
x_steps = np.random.normal(loc=0, scale=step_size, size=num_steps)
y_steps = np.random.normal(loc=0, scale=step_size, size=num_steps)

# 3. Cumulatively add the steps to get the actual positions over time
# [0, 0] ensures the particle starts at the origin
x_positions = np.concatenate(([0], np.cumsum(x_steps)))
y_positions = np.concatenate(([0], np.cumsum(y_steps)))

# 4. Plot the results
plt.figure(figsize=(8, 8))
plt.plot(x_positions, y_positions, label="Particle Path", color="blue", alpha=0.7, linewidth=1)

# Mark the start and end points
plt.scatter(x_positions[0], y_positions[0], color="green", s=100, label="Start (0,0)", zorder=5)
plt.scatter(x_positions[-1], y_positions[-1], color="red", s=100, label="End", zorder=5)

# Aesthetics
plt.title(f"2D Brownian Motion Simulation ({num_steps} Steps)", fontsize=14)
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.axis("equal")  # Keeps the scale uniform on both axes

# Show the plot
plt.show()
