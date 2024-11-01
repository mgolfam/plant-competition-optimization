import numpy as np
import math  # Import the math module

class MarinePredatorsAlgorithm:
    def __init__(self, lb, ub, dim, population_size, max_iterations, fobj):
        """
        Initialize the Marine Predators Algorithm (MPA).
        """
        self.lb = lb if isinstance(lb, list) else [lb] * dim
        self.ub = ub if isinstance(ub, list) else [ub] * dim
        self.dim = dim
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.fobj = fobj

        # Initialize positions of predators
        self.population = np.random.uniform(0, 1, (population_size, dim)) * (np.array(self.ub) - np.array(self.lb)) + np.array(self.lb)
        self.fitness = np.full(population_size, np.inf)

        # Initialize best predator (best solution)
        self.best_position = np.zeros(dim)
        self.best_fitness = float('inf')

        # Track the convergence
        self.convergence_curve = np.zeros(max_iterations)

    def levy(self):
        """
        Levy flight mechanism to generate random steps.
        """
        beta = 1.5
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
                 (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / np.abs(v) ** (1 / beta)
        return step

    def run(self):
        """
        Execute the Marine Predators Algorithm (MPA).
        """
        iteration = 0
        for iteration in range(self.max_iterations):
            # Update the positions of the population
            for i in range(self.population_size):
                r1 = np.random.rand()
                r2 = np.random.rand()

                if r1 < 0.5:
                    # Eq. (3.4) in MPA paper
                    self.population[i, :] = self.population[i, :] + r1 * (self.population[i, :] - r2 * self.population[i, :])

                # Boundary control
                self.population[i, :] = np.clip(self.population[i, :], self.lb, self.ub)

                # Fitness evaluation
                self.fitness[i] = self.fobj(self.population[i, :])

                # Update the best solution
                if self.fitness[i] < self.best_fitness:
                    self.best_fitness = self.fitness[i]
                    self.best_position = np.copy(self.population[i, :])

            # Phase 2: Low prey density, intensification using Levy flight
            for i in range(self.population_size):
                self.population[i, :] = self.best_position + 0.1 * self.levy() * (self.population[i, :] - self.best_position)

                # Boundary control
                self.population[i, :] = np.clip(self.population[i, :], self.lb, self.ub)

                # Fitness evaluation
                self.fitness[i] = self.fobj(self.population[i, :])

                if self.fitness[i] < self.best_fitness:
                    self.best_fitness = self.fitness[i]
                    self.best_position = np.copy(self.population[i, :])

            # Track convergence
            self.convergence_curve[iteration] = self.best_fitness

            # Display progress
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Best Fitness: {self.best_fitness}")

        return self.best_position, self.best_fitness, self.convergence_curve


# Example objective function (e.g., Sphere function)
def sphere_function(x):
    return np.sum(x ** 2)

# Example usage
if __name__ == "__main__":
    lb = -10  # Lower bound
    ub = 10  # Upper bound
    dim = 30  # Dimensionality of the problem
    population_size = 50  # Number of predators
    max_iterations = 100  # Number of iterations

    # Instantiate the MPA algorithm
    mpa = MarinePredatorsAlgorithm(lb, ub, dim, population_size, max_iterations, sphere_function)
    best_position, best_fitness, convergence_curve = mpa.run()

    print(f"Best Position: {best_position}")
    print(f"Best Fitness: {best_fitness}")
