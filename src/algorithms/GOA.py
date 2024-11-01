import numpy as np

class GrasshopperOptimizationAlgorithm:
    def __init__(self, A, B, population_size, max_iterations, dim, fobj):
        """
        Initialize the Grasshopper Optimization Algorithm (GOA).
        
        Parameters:
        A (float): Lower bound of the search space.
        B (float): Upper bound of the search space.
        population_size (int): Number of grasshoppers in the population.
        max_iterations (int): Maximum number of iterations.
        dim (int): Dimensionality of the problem.
        fobj (function): Objective function to minimize.
        """
        self.A = A
        self.B = B
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.dim = dim
        self.fobj = fobj
        
        # Initialize the population of grasshoppers with random positions
        self.population = A + (B - A) * np.random.rand(population_size, dim)
        self.best_solution = None
        self.best_fitness = float('inf')

        # Coefficients for controlling the influence of neighbors
        self.c_max = 1.0  # Maximum attraction coefficient
        self.c_min = 0.00001  # Minimum attraction coefficient

    def _fitness(self, population):
        # Evaluate the fitness of the population
        return np.array([self.fobj(ind) for ind in population])

    def _compute_attraction(self, current_grasshopper, target_grasshopper, c):
        # Compute the attraction between two grasshoppers based on distance
        distance = np.linalg.norm(target_grasshopper - current_grasshopper)
        s = 2 * np.exp(-distance / c) * np.sin(np.pi * distance / c)
        return s * (target_grasshopper - current_grasshopper) / distance

    def _update_position(self, grasshopper, population, c):
        # Update the position of the grasshopper based on the attraction of other grasshoppers
        new_position = np.zeros(self.dim)
        for j in range(self.population_size):
            if not np.array_equal(grasshopper, population[j]):
                new_position += self._compute_attraction(grasshopper, population[j], c)
        return np.clip(grasshopper + new_position, self.A, self.B)

    def run(self):
        """
        Execute the Grasshopper Optimization Algorithm.
        
        Returns:
        tuple: Best solution found and its fitness.
        """
        for iteration in range(self.max_iterations):
            # Compute the attraction coefficient (c) that decreases over iterations
            c = self.c_max - iteration * ((self.c_max - self.c_min) / self.max_iterations)
            
            # Evaluate fitness of the population
            fitness = self._fitness(self.population)
            
            # Update the best solution found so far
            min_fitness_idx = np.argmin(fitness)
            if fitness[min_fitness_idx] < self.best_fitness:
                self.best_fitness = fitness[min_fitness_idx]
                self.best_solution = self.population[min_fitness_idx].copy()
            
            # Update the position of each grasshopper
            new_population = np.zeros_like(self.population)
            for i in range(self.population_size):
                new_population[i] = self._update_position(self.population[i], self.population, c)
            
            self.population = new_population
        
        return self.best_solution, self.best_fitness


# Sample Objective Function (e.g., Sphere function)
def sphere_function(x):
    return np.sum(x**2)

# Example usage
if __name__ == "__main__":
    A = -10
    B = 10
    population_size = 30  # Number of grasshoppers
    max_iterations = 100  # Number of iterations
    dim = 5  # Dimensionality of the problem

    goa = GrasshopperOptimizationAlgorithm(A, B, population_size, max_iterations, dim, sphere_function)
    best_solution, best_fitness = goa.run()

    # Output the result
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness: {best_fitness}")
