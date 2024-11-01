import numpy as np

class WhaleOptimizationAlgorithm:
    def __init__(self, lb, ub, dim, population_size, max_iterations, fobj):
        """
        Initialize the Whale Optimization Algorithm (WOA).

        Parameters:
        lb (float or list): Lower bound of the search space.
        ub (float or list): Upper bound of the search space.
        dim (int): Dimensionality of the problem.
        population_size (int): Number of whales in the population.
        max_iterations (int): Maximum number of iterations.
        fobj (function): Objective function to minimize.
        """
        self.lb = lb if isinstance(lb, list) else [lb] * dim
        self.ub = ub if isinstance(ub, list) else [ub] * dim
        self.dim = dim
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.fobj = fobj

        # Initialize positions of the whales
        self.population = np.random.uniform(0, 1, (population_size, dim)) * (np.array(self.ub) - np.array(self.lb)) + np.array(self.lb)
        self.fitness = np.full(population_size, np.inf)

        # Initialize leader whale (best solution)
        self.leader_position = np.zeros(dim)
        self.leader_fitness = float('inf')

        # Track the convergence
        self.convergence_curve = np.zeros(max_iterations)

    def run(self):
        """
        Execute the Whale Optimization Algorithm (WOA).

        Returns:
        tuple: Best solution found and its fitness.
        """
        for iteration in range(self.max_iterations):
            # Linearly decrease 'a' from 2 to 0
            a = 2 - iteration * (2 / self.max_iterations)

            # Update each whale's position
            for i in range(self.population_size):
                r1 = np.random.rand()
                r2 = np.random.rand()

                A = 2 * a * r1 - a  # Equation (2.3) in the paper
                C = 2 * r2  # Equation (2.4)

                p = np.random.rand()
                b = 1  # Constant in spiral equation
                l = np.random.uniform(-1, 1)  # Random number between -1 and 1

                distance_to_leader = np.abs(C * self.leader_position - self.population[i, :])

                if p < 0.5:
                    if np.abs(A) < 1:
                        # Shrinking encircling mechanism
                        self.population[i, :] = self.leader_position - A * distance_to_leader
                    else:
                        # Search for prey (random whale)
                        random_whale = np.random.randint(0, self.population_size)
                        distance_to_random = np.abs(C * self.population[random_whale, :] - self.population[i, :])
                        self.population[i, :] = self.population[random_whale, :] - A * distance_to_random
                else:
                    # Spiral updating position (Equation 2.5)
                    distance_to_leader = np.abs(self.leader_position - self.population[i, :])
                    self.population[i, :] = distance_to_leader * np.exp(b * l) * np.cos(2 * np.pi * l) + self.leader_position

                # Boundary checking
                self.population[i, :] = np.clip(self.population[i, :], self.lb, self.ub)

            # Evaluate the fitness of each whale
            for i in range(self.population_size):
                self.fitness[i] = self.fobj(self.population[i, :])

                # Update leader whale if a better solution is found
                if self.fitness[i] < self.leader_fitness:
                    self.leader_fitness = self.fitness[i]
                    self.leader_position = np.copy(self.population[i, :])

            # Track convergence
            self.convergence_curve[iteration] = self.leader_fitness

            # Output the progress every few iterations
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Best Fitness: {self.leader_fitness}")

        return self.leader_position, self.leader_fitness, self.convergence_curve


# Example objective function (e.g., Sphere function)
def sphere_function(x):
    return np.sum(x ** 2)

# Example usage
if __name__ == "__main__":
    lb = -10  # Lower bound
    ub = 10  # Upper bound
    dim = 30  # Dimensionality of the problem
    population_size = 50  # Number of whales
    max_iterations = 100  # Number of iterations

    # Instantiate the WOA algorithm
    woa = WhaleOptimizationAlgorithm(lb, ub, dim, population_size, max_iterations, sphere_function)
    best_position, best_fitness, convergence_curve = woa.run()

    print(f"Best Position: {best_position}")
    print(f"Best Fitness: {best_fitness}")
