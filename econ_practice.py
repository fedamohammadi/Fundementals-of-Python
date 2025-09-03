import numpy as np
import matplotlib.pyplot as plt

# Apples (x-axis)
A = np.linspace(1, 100, 500)
# Bananas from the indifference curve
B = 1350 / A

# Plot curve
plt.plot(A, B, label="Indifference curve U=1350")
plt.scatter(15, 90, color="red", zorder=5, label="Current bundle (15,90)")

# Add tangent line at (15,90) with slope -6
slope = -6
x0, y0 = 15, 90
tangent_x = np.linspace(5, 25, 100)
tangent_y = slope * (tangent_x - x0) + y0
plt.plot(tangent_x, tangent_y, 'g--', label="Tangent slope -6")

plt.xlabel("Apples (A)")
plt.ylabel("Bananas (B)")
plt.title("Maisie’s Indifference Curve")
plt.legend()
plt.grid(True)
plt.show()
