import numpy as np

class BenchmarkFunctions:
    def __init__(self, search_domain, num_local_minima, global_minimum):
        self.search_domain = np.array(search_domain)
        self.num_local_minima = num_local_minima
        self.global_minimum = global_minimum

    def enforce_domain(self, x):
        """Ensure x is within the specified search domain."""
        return np.clip(x, self.search_domain[0], self.search_domain[1])

    # 1. Branin Function
    def branin(self, x):
        x = self.enforce_domain(x)
        a = 1.0
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        r = 6.0
        s = 10.0
        t = 1 / (8 * np.pi)
        return a * (x[1] - b * x[0]**2 + c * x[0] - r)**2 + s * (1 - t) * np.cos(x[0]) + s

    # 2. Easom Function
    def easom(self, x):
        x = self.enforce_domain(x)
        return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-(x[0] - np.pi)**2 - (x[1] - np.pi)**2)

    # 3. Goldstein & Price Function
    def goldstein_price(self, x):
        x = self.enforce_domain(x)
        return (1 + (x[0] + x[1] + 1)**2 * (19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2)) * \
               (30 + (2*x[0] - 3*x[1])**2 * (18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2))

    # 4. Six-Hump Camel Back Function
    def six_hump_camel_back(self, x):
        x = self.enforce_domain(x)
        term1 = (4 - 2.1 * x[0]**2 + (x[0]**4) / 3) * x[0]**2
        term2 = x[0] * x[1]
        term3 = (-4 + 4 * x[1]**2) * x[1]**2
        return term1 + term2 + term3

    # 5. Shubert Function
    def shubert(self, x):
        x = self.enforce_domain(x)
        sum1 = sum(i * np.cos((i + 1) * x[0] + i) for i in range(1, 6))
        sum2 = sum(i * np.cos((i + 1) * x[1] + i) for i in range(1, 6))
        return sum1 * sum2

    # 6. Shekel Function
    def shekel(self, x):
        x = self.enforce_domain(x)
        m = 10
        a = np.array([[4.0, 4.0, 4.0, 4.0],
                      [1.0, 1.0, 1.0, 1.0],
                      [8.0, 8.0, 8.0, 8.0],
                      [6.0, 6.0, 6.0, 6.0],
                      [3.0, 7.0, 3.0, 7.0],
                      [2.0, 9.0, 2.0, 9.0],
                      [5.0, 5.0, 3.0, 3.0],
                      [8.0, 1.0, 8.0, 1.0],
                      [6.0, 2.0, 6.0, 2.0],
                      [7.0, 3.6, 7.0, 3.6]])
        c = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
        return -np.sum([1.0 / (np.sum((x - a[i])**2) + c[i]) for i in range(m)])

    # 7. Schwefel Function
    def schwefel(self, x):
        x = self.enforce_domain(x)
        return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

    # 8. Griewank Function
    def griewank(self, x):
        x = self.enforce_domain(x)
        return 1 + np.sum(x**2 / 4000) - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))

    # 9. Rastrigin Function
    def rastrigin(self, x):
        x = self.enforce_domain(x)
        return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

    # 10. Ackley Function
    def ackley(self, x):
        x = self.enforce_domain(x)
        return -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2))) - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.e

    # 11. Beale Function
    def beale(self, x):
        x = self.enforce_domain(x)
        return (1.5 - x[0] + x[0] * x[1])**2 + (2.25 - x[0] + x[0] * x[1]**2)**2 + (2.625 - x[0] + x[0] * x[1]**3)**2

    # 12. Bohachevsky Function
    def bohachevsky(self, x):
        x = self.enforce_domain(x)
        return x[0]**2 + 2 * x[1]**2 - 0.3 * np.cos(3 * np.pi * x[0]) - 0.4 * np.cos(4 * np.pi * x[1]) + 0.7

    # 13. Booth Function
    def booth(self, x):
        x = self.enforce_domain(x)
        return (x[0] + 2 * x[1] - 7)**2 + (2 * x[0] + x[1] - 5)**2

    # 14. Colville Function
    def colville(self, x):
        x = self.enforce_domain(x)
        return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2 + 90 * (x[3] - x[2]**2)**2 + (1 - x[2])**2 + \
               10.1 * ((x[1] - 1)**2 + (x[3] - 1)**2) + 19.8 * (x[1] - 1) * (x[3] - 1)

    # 15. Himmelblau Function
    def himmelblau(self, x):
        x = self.enforce_domain(x)
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

    # 16. Three-Hump Camel Function
    def three_hump_camel(self, x):
        x = self.enforce_domain(x)
        return 2 * x[0]**2 - 1.05 * x[0]**4 + (x[0]**6) / 6 + x[0] * x[1] + x[1]**2

    # 17. Matyas Function
    def matyas(self, x):
        x = self.enforce_domain(x)
        return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]

    # 18. Hartmann 3 Function
    def hartmann_3(self, x):
        x = self.enforce_domain(x)
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([[3.0, 10, 30],
                      [0.1, 10, 35],
                      [3.0, 10, 30],
                      [0.1, 10, 35]])
        P = 10**-4 * np.array([[3689, 1170, 2673],
                               [4699, 4387, 7470],
                               [1091, 8732, 5547],
                               [381, 5743, 8828]])
        return -np.sum([alpha[i] * np.exp(-np.sum(A[i] * (x - P[i])**2)) for i in range(4)])

    # 19. Hartmann 6 Function
    def hartmann_6(self, x):
        x = self.enforce_domain(x)
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                      [0.05, 10, 17, 0.1, 8, 14],
                      [3.0, 3.5, 1.7, 10, 17, 8],
                      [17, 8, 0.05, 10, 0.1, 14]])
        P = 10**-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                               [2329, 4135, 8307, 3736, 1004, 9991],
                               [2348, 1415, 3522, 2883, 3047, 6650],
                               [4047, 8828, 8732, 5743, 1091, 381]])
        return -np.sum([alpha[i] * np.exp(-np.sum(A[i] * (x - P[i])**2)) for i in range(4)])

    # 20. Schaffer Function
    def schaffer_function(self, x):
        x = self.enforce_domain(x)
        x1, x2 = x[0], x[1]
        numerator = (np.sin(np.sqrt(x1**2 + x2**2)))**2 - 0.5
        denominator = (1 + 0.001 * (x1**2 + x2**2))**2
        return 0.5 + numerator / denominator

# Example Usage
search_domain = [(-5, -5), (10, 15)]  # Example search domain for 2D functions
functions = BenchmarkFunctions(search_domain, num_local_minima=5, global_minimum=(1, 1))

x = np.array([1.0, 2.0])  # Example input
print("Branin Function Value:", functions.branin(x))
print("Shekel Function Value:", functions.shekel(x))

benchmark_configs = [
    {
        "name": "Branin",
        "search_domain": [(-5, 10), (0, 15)],
        "num_local_minima": 0,
        "global_minimum": [(np.pi, 2.275), (9.42478, 2.475), (-3.14159, 12.275)],
        "global_minimum_f": [0.397887],
    },
    {
        "name": "Easom",
        "search_domain": [(-100, 100), (-100, 100)],
        "num_local_minima": "Several",
        "global_minimum": [(np.pi, np.pi)],
        "global_minimum_f": [-1],
    },
    {
        "name": "Goldstein & Price",
        "search_domain": [(-2, 2), (-2, 2)],
        "num_local_minima": "Several",
        "global_minimum": [(0, -1)],
        "global_minimum_f": [3],
    },
    {
        "name": "Six-Hump Camel Back",
        "search_domain": [(-3, -2), (3, 2)],
        "num_local_minima": 6,
        "global_minimum": [(-0.0898, 0.7126), (0.0898, -0.7126)],
        "global_minimum_f": [-1.031628453],
    },
    {
        "name": "Shubert",
        "search_domain": [(-10, 10), (-10, 10)],
        "num_local_minima": "several",
        "global_minimum": "18 global minimal",
        "global_minimum_f": [-186.7309],
    },
    {
        "name": "Shekel",
        "search_domain": [(0, 10), (0, 10), (0, 10), (0, 10)],
        "num_local_minima": "m local",
        "global_minimum": [4,4,4,4],
        "global_minimum_f": [-10.1532, -10.4029, -10.5364],
    },
    {
        "name": "Schwefel",
        "search_domain": [(-500, 500), -(500, 500)],
        "num_local_minima": "Many",
        "global_minimum": "420.9687 * n with f(x*) = 0",
        "global_minimum_f": [0],
    },
    {
        "name": "Griewank",
        "search_domain": [(-600, 600), (-600, 600)],
        "num_local_minima": "Many",
        "global_minimum": [(0, 0)],
        "global_minimum_f": [0],
    },
    {
        "name": "Rastrigin",
        "search_domain": [(-5.12, -5.12), (5.12, 5.12)],
        "num_local_minima": "Many",
        "global_minimum": [(0, 0)]
    },
    {
        "name": "Ackley",
        "search_domain": [(-15, 30), (32.768, 32.768)],
        "num_local_minima": "Many",
        "global_minimum": [(0, 0)],
        "global_minimum_f": [0],
    },
    {
        "name": "Beale",
        "search_domain": [(-4.5, 4.5), (-4.5, 4.5)],
        "num_local_minima": "Several",
        "global_minimum": [(3, 0.5)],
        "global_minimum_f": [0],
    },
    {
        "name": "Bohachevsky",
        "search_domain": [(-100, -100), (100, 100)],
        "num_local_minima": "None",
        "global_minimum": [(0, 0)]
    },
    {
        "name": "Booth",
        "search_domain": [(-10, -10), (10, 10)],
        "num_local_minima": "None",
        "global_minimum": [(1, 3)]
    },
    {
        "name": "Colville",
        "search_domain": [(-10, 10), (-10, 10)],
        "num_local_minima": "0",
        "global_minimum": [(0, 0)],
        "global_minimum_f": [0],
    },
    {
        "name": "Himmelblau",
        "search_domain": [(-6, 6), (-6, 6)],
        "num_local_minima": "0",
        "global_minimum": [(3, 2), (-2.8051, 3.1313), (-3.7793, -3.2831), (3.5844, -1.8481)],
        "global_minimum_f": [0],
    },
    {
        "name": "Hump",
        "search_domain": [(-5, 5), (-5, 5)],
        "num_local_minima": "None",
        "global_minimum": [(0.0898, 0.7126), (0.0898, 0.7126)],
        "global_minimum_f": [0],
    },
    {
        "name": "Matyas",
        "search_domain": [(-10, 10), (-10, 10)],
        "num_local_minima": "None",
        "global_minimum": [(0, 0)],
        "global_minimum_f": [0],
    },
    {
        "name": "Hartmann 3",
        "search_domain": [(0, 1, (0, 1), (0, 1))],
        "num_local_minima": "Multiple",
        "global_minimum": [(0.114614, 0.555649, 0.852547)],
        "global_minimum_f": [-3.86278],
    },
    {
        "name": "Hartmann 6",
        "search_domain": [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
        "num_local_minima": "Multiple",
        "global_minimum": [(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)],
        "global_minimum_f": [0],
    },
    {
        "name": "Schaffer",
        "search_domain": [(-100, -100), (100, 100)],
        "num_local_minima": "Many",
        "global_minimum": [(0, 0)],
        "global_minimum_f": [0],
    }
]

