import numpy as np

class GrayWolfOptimization:
    def __init__(self, fitness_function, num_wolves=30, max_iterations=100, dimensions=2, a_init=2):
        """
        Initialize the GWO algorithm.

        :param fitness_function: Objective function to minimize
        :param num_wolves: Number of wolves in the pack
        :param max_iterations: Maximum number of iterations to run the algorithm
        :param dimensions: Number of dimensions of the search space
        :param a_init: Initial value of 'a' parameter (reduces to 0 over iterations)
        """
        self.fitness_function = fitness_function
        self.num_wolves = num_wolves
        self.max_iterations = max_iterations
        self.dimensions = dimensions
        self.a_init = a_init
        
        # Initialize positions of wolves randomly in search space
        self.wolves_positions = np.random.uniform(-10, 10, (self.num_wolves, self.dimensions))
        self.alpha_position, self.beta_position, self.delta_position = np.zeros(dimensions), np.zeros(dimensions), np.zeros(dimensions)
        self.alpha_score, self.beta_score, self.delta_score = float("inf"), float("inf"), float("inf")
        
    def optimize(self):
        """
        Run the GWO optimization process and return the best position and fitness value found.
        """
        for iteration in range(self.max_iterations):
            a = self.a_init - (iteration * self.a_init / self.max_iterations)  # Reduce 'a' linearly from a_init to 0
            
            # Evaluate fitness for each wolf and update alpha, beta, delta
            for i in range(self.num_wolves):
                fitness = self.fitness_function(self.wolves_positions[i])
                if fitness < self.alpha_score:
                    self.alpha_score, self.alpha_position = fitness, self.wolves_positions[i].copy()
                elif fitness < self.beta_score:
                    self.beta_score, self.beta_position = fitness, self.wolves_positions[i].copy()
                elif fitness < self.delta_score:
                    self.delta_score, self.delta_position = fitness, self.wolves_positions[i].copy()
            
            # Update positions of all wolves based on alpha, beta, delta
            self._update_wolves_positions(a)
            
            # Optional: Logging for each iteration
            print(f"Iteration {iteration+1}: Alpha fitness = {self.alpha_score}")

        return self.alpha_position, self.alpha_score
    
    def _update_wolves_positions(self, a):
        """
        Update the positions of the wolves based on alpha, beta, delta wolves.
        :param a: Current scaling factor for exploration-exploitation balance
        """
        for i in range(self.num_wolves):
            # Calculate distances to alpha, beta, delta wolves
            D_alpha = np.abs(2 * a * np.random.rand(self.dimensions) * self.alpha_position - self.wolves_positions[i])
            D_beta = np.abs(2 * a * np.random.rand(self.dimensions) * self.beta_position - self.wolves_positions[i])
            D_delta = np.abs(2 * a * np.random.rand(self.dimensions) * self.delta_position - self.wolves_positions[i])
            
            # Calculate new positions based on alpha, beta, delta
            X1 = self.alpha_position - a * D_alpha
            X2 = self.beta_position - a * D_beta
            X3 = self.delta_position - a * D_delta
            
            # Average to determine new position of the wolf
            self.wolves_positions[i] = (X1 + X2 + X3) / 3
