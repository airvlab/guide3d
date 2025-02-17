import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as si

# Given data points
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 1, 0, 1, 0, 1])

# Degree of the B-spline
degree = 3

# Given knots (must be in non-decreasing order)
knots = np.linspace(0, 5, 10)

# Solve for control points using least squares fitting
tck = si.make_lsq_spline(x, y, knots, degree)

# Generate smooth curve
x_smooth = np.linspace(0, 5, 100)
y_smooth = tck(x_smooth)

# Plot results
plt.scatter(x, y, color="red", label="Data Points")
plt.plot(x_smooth, y_smooth, label="Fitted B-Spline", linewidth=2)
plt.legend()
plt.show()
