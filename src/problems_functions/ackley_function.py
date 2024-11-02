import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D


# Define the 2D Ackley function
def ackley_function(x, y):
    a = 20
    b = 0.2
    c = 2 * np.pi
    sum_sq_term = -b * np.sqrt(0.5 * (x**2 + y**2))
    cos_term = 0.5 * (np.cos(c * x) + np.cos(c * y))
    return -a * np.exp(sum_sq_term) - np.exp(cos_term) + a + np.exp(1)

def plot_3d_ackley():
    # Generate a grid of points in the range [-5, 5] for both x and y
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = ackley_function(X, Y)

    # Create a 3D plot of the 2D Ackley function
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Function Value')
    ax.set_title('3D Ackley Function')
    plt.show()

# Call the function to plot
if __name__ == "__main__":
    plot_3d_ackley()
