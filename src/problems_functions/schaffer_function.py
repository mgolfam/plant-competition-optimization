import numpy as np

# Define Schaffer's function
def schaffer_function(x, y):
    numerator = (np.sin(np.sqrt(x**2 + y**2)))**2 - 0.5
    denominator = (1 + 0.001 * (x**2 + y**2))**2
    return 0.5 + numerator / denominator

# Example usage
x = np.linspace(-100, 100, 400)
y = np.linspace(-100, 100, 400)
X, Y = np.meshgrid(x, y)
Z = schaffer_function(X, Y)
