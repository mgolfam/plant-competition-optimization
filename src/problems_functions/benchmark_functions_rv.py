import numpy as np

class OptimizationFunction:
    def __init__(self, search_domain, num_local_minima, global_minimum):
        self.search_domain = search_domain
        self.num_local_minima = num_local_minima
        self.global_minimum = global_minimum

    def enforce_domain(self, x):
        """Ensure x is within the specified search domain."""
        return np.clip(x, self.search_domain[0], self.search_domain[1])

    def branin(self, x):
        x = self.enforce_domain(x)
        x1, x2 = x[0], x[1]
        a = 1
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)
        return a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s

    def easom(self, x):
        x = self.enforce_domain(x)
        x1, x2 = x[0], x[1]
        return -np.cos(x1) * np.cos(x2) * np.exp(-((x1 - np.pi)**2 + (x2 - np.pi)**2))

    def goldstein_price(self, x):
        x = self.enforce_domain(x)
        x1, x2 = x[0], x[1]
        part1 = 1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)
        part2 = 30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2)
        return part1 * part2

    def six_hump_camel_back(self, x):
        x = self.enforce_domain(x)
        x1, x2 = x[0], x[1]
        return (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2 + x1 * x2 + (-4 + 4 * x2**2) * x2**2

    def shubert(self, x):
        x = self.enforce_domain(x)
        x1, x2 = x[0], x[1]
        sum1 = sum(i * np.cos((i + 1) * x1 + i) for i in range(1, 6))
        sum2 = sum(i * np.cos((i + 1) * x2 + i) for i in range(1, 6))
        return sum1 * sum2

    def schwefel(self, x):
        x = self.enforce_domain(x)
        return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

    def griewank(self, x):
        x = self.enforce_domain(x)
        sum_part = np.sum(x**2) / 4000
        prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return sum_part - prod_part + 1

    def ackley(self, x):
        x = self.enforce_domain(x)
        n = len(x)
        return -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e

    def sphere(self, x):
        x = self.enforce_domain(x)
        return np.sum(x**2)

    def schaffer(self, x):
        x = self.enforce_domain(x)
        x1, x2 = x[0], x[1]
        numerator = (np.sin(np.sqrt(x1**2 + x2**2)))**2 - 0.5
        denominator = (1 + 0.001 * (x1**2 + x2**2))**2
        return 0.5 + numerator / denominator

    def shekel(self, x, a, c):
        x = self.enforce_domain(x)
        m = a.shape[1]  # Number of maxima
        result = -np.sum(1 / (np.sum((x - a[:, j])**2) + c[j]) for j in range(m))
        return result

# Example Usage
shekel_function = OptimizationFunction(
    search_domain=(np.array([-10] * 4), np.array([10] * 4)),  # Example for 4 dimensions
    num_local_minima=10,
    global_minimum=np.array([4, 4, 4, 4])
)

# Define parameters for the Shekel function
a = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6]]).T
c = np.array([0.1, 0.2, 0.4, 0.8])

x = np.array([4, 4, 4, 4])  # Example input
print("Shekel Function Value:", shekel_function.shekel(x, a, c))
