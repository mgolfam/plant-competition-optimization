import numpy as np

# 1. Branin Function
def branin(x):
    a = 1.0
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6.0
    s = 10.0
    t = 1 / (8 * np.pi)
    return a * (x[1] - b * x[0]**2 + c * x[0] - r)**2 + s * (1 - t) * np.cos(x[0]) + s

# 2. Easom Function
def easom(x):
    return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-(x[0] - np.pi)**2 - (x[1] - np.pi)**2)

# 3. Goldstein & Price Function
def goldstein_price(x):
    return (1 + (x[0] + x[1] + 1)**2 * (19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2)) * \
           (30 + (2*x[0] - 3*x[1])**2 * (18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2))

# 4. Six-Hump Camel Back Function
def six_hump_camel_back(x):
    term1 = (4 - 2.1 * x[0]**2 + (x[0]**4) / 3) * x[0]**2
    term2 = x[0] * x[1]
    term3 = (-4 + 4 * x[1]**2) * x[1]**2
    return term1 + term2 + term3

# 5. Shubert Function
def shubert(x):
    sum1 = 0
    sum2 = 0
    for i in range(1, 6):
        sum1 += i * np.cos((i + 1) * x[0] + i)
        sum2 += i * np.cos((i + 1) * x[1] + i)
    return sum1 * sum2

# 6. Shekel Function
def shekel(x):
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
def schwefel(x):
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

# 8. Griewank Function
def griewank(x):
    return 1 + np.sum(x**2 / 4000) - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))

# 9. Rastrigin Function
def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

# 10. Ackley Function
def ackley(x):
    return -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2))) - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.e

# 11. Beale Function
def beale(x):
    return (1.5 - x[0] + x[0] * x[1])**2 + (2.25 - x[0] + x[0] * x[1]**2)**2 + (2.625 - x[0] + x[0] * x[1]**3)**2

# 12. Bohachevsky Function
def bohachevsky(x):
    return x[0]**2 + 2 * x[1]**2 - 0.3 * np.cos(3 * np.pi * x[0]) - 0.4 * np.cos(4 * np.pi * x[1]) + 0.7

# 13. Booth Function
def booth(x):
    return (x[0] + 2 * x[1] - 7)**2 + (2 * x[0] + x[1] - 5)**2

# 14. Colville Function
def colville(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2 + 90 * (x[3] - x[2]**2)**2 + (1 - x[2])**2 + 10.1 * ((x[1] - 1)**2 + (x[3] - 1)**2) + 19.8 * (x[1] - 1) * (x[3] - 1)

# 15. Himmelblau Function
def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

# 16. Three-Hump Camel Function
def three_hump_camel(x):
    return 2 * x[0]**2 - 1.05 * x[0]**4 + (x[0]**6) / 6 + x[0] * x[1] + x[1]**2

# 17. Matyas Function
def matyas(x):
    return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]

# 18. Hartmann 3 Function
def hartmann_3(x):
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
def hartmann_6(x):
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

# 20. Ackley Function
def ackley(x):
    return -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2))) - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.e

# 21. Sphere Function
def sphere(x):
    return np.sum(x**2)

def schaffer_function(x):
    # Assuming x is a 1D array with two elements: [x1, x2]
    x1, x2 = x[0], x[1]
    numerator = (np.sin(np.sqrt(x1**2 + x2**2)))**2 - 0.5
    denominator = (1 + 0.001 * (x1**2 + x2**2))**2
    return 0.5 + numerator / denominator