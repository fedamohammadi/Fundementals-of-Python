import numpy as np
import matplotlib.pyplot as plt
import os

# Optional: make sure the Figures folder exists
os.makedirs("Figures", exist_ok=True)

# 1. Generate example data
np.random.seed(0)
x = np.random.uniform(0, 10, 50)              # Hours studied
y = 50 + 3 * x + np.random.normal(0, 4, 50)   # Exam score with noise

# 2. Fit a simple linear regression line using numpy.polyfit
#    y ≈ m*x + b
m, b = np.polyfit(x, y, 1)

# 3. Create smooth line for plotting the regression line
x_line = np.linspace(x.min(), x.max(), 100)
y_line = m * x_line + b

# 4. Plot scatter + regression line
plt.figure(figsize=(7, 5))
plt.scatter(x, y, label="Data points")
plt.plot(x_line, y_line, label="Regression line")

plt.xlabel("Hours studied")
plt.ylabel("Exam score")
plt.title("Scatterplot with Regression Line")
plt.legend()
plt.tight_layout()

