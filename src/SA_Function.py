import numpy as np

class SimulatedAnnealing:
    def __init__(self, A, B, initial_temperature, cooling_rate, max_iterations, dim, fobj):
        """
        Initialize the Simulated Annealing (SA) algorithm.
        
        Parameters:
        A (float): Lower bound of the search space.
        B (float): Upper bound of the search space.
        initial_temperature (float): Starting temperature.
        cooling_rate (float): Rate at which the temperature decreases.
        max_iterations (int): Maximum number of iterations.
        dim (int): Dimensionality of the problem.
        fobj (function): Objective function to minimize.
        """
        self.A = A
        self.B = B
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.dim = dim
        self.fobj = fobj
        
        # Initialize the current solution with random values
        self.current_solution = A + (B - A) * np.random.rand(dim)
        self.best_solution = self.current_solution.copy()
        self.current_fitness = fobj(self.current_solution)
        self.best_fitness = self.current_fitness

    def _acceptance_probability(self, new_fitness):
        # If the new solution is better, accept it
        if new_fitness < self.current_fitness:
            return 1.0
        # Otherwise, accept it with a probability that depends on the temperature
        return np.exp((self.current_fitness - new_fitness) / self.temperature)

    def _generate_neighbor(self):
        # Generate a new solution by modifying the current solution slightly
        neighbor = self.current_solution + np.random.uniform(-1, 1, self.dim)
        # Ensure the neighbor is within bounds
        neighbor = np.clip(neighbor, self.A, self.B)
        return neighbor

    def run(self):
        """
        Execute the Simulated Annealing algorithm.
        
        Returns:
        tuple: Best solution found and its fitness.
        """
        for iteration in range(self.max_iterations):
            # Generate a neighboring solution
            new_solution = self._generate_neighbor()
            new_fitness = self.fobj(new_solution)
            
            # Decide whether to accept the new solution
            if np.random.rand() < self._acceptance_probability(new_fitness):
                self.current_solution = new_solution
                self.current_fitness = new_fitness
            
            # Update the best solution if the current solution is better
            if self.current_fitness < self.best_fitness:
                self.best_solution = self.current_solution.copy()
                self.best_fitness = self.current_fitness
            
            # Cool down the temperature
            self.temperature *= self.cooling_rate
        
        return self.best_solution, self.best_fitness


# Sample Objective Function (e.g., Sphere function)
def sphere_function(x):
    return np.sum(x**2)

# Example usage
if __name__ == "__main__":
    A = -10
    B = 10
    initial_temperature = 1000
    cooling_rate = 0.95
    max_iterations = 500
    dim = 5  # Dimensionality of the problem

    sa = SimulatedAnnealing(A, B, initial_temperature, cooling_rate, max_iterations, dim, sphere_function)
    best_solution, best_fitness = sa.run()

    # Output the result
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness: {best_fitness}")
