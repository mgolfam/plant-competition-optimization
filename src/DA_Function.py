import numpy as np

class DragonflyAlgorithm:
    def __init__(self, A, B, population_size, max_iterations, dim, fobj, w=0.9, s=0.1, a=0.1, c=0.1, f=0.1, e=0.1):
        """
        Initialize the Dragonfly Algorithm (DA).
        
        Parameters:
        A (float): Lower bound of the search space.
        B (float): Upper bound of the search space.
        population_size (int): Number of dragonflies in the population.
        max_iterations (int): Maximum number of iterations.
        dim (int): Dimensionality of the problem.
        fobj (function): Objective function to minimize.
        w (float): Inertia weight.
        s (float): Separation weight.
        a (float): Alignment weight.
        c (float): Cohesion weight.
        f (float): Attraction weight (food).
        e (float): Distraction weight (enemy).
        """
        self.A = A
        self.B = B
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.dim = dim
        self.fobj = fobj
        self.w = w  # Inertia weight
        self.s = s  # Separation weight
        self.a = a  # Alignment weight
        self.c = c  # Cohesion weight
        self.f = f  # Food attraction weight
        self.e = e  # Enemy distraction weight
        
        # Initialize the population of dragonflies
        self.population = A + (B - A) * np.random.rand(population_size, dim)
        self.velocity = np.zeros((population_size, dim))  # Initialize velocities
        self.best_solution = None
        self.best_fitness = float('inf')
        
        # Initialize food and enemy positions randomly
        self.food_position = A + (B - A) * np.random.rand(dim)
        self.enemy_position = A + (B - A) * np.random.rand(dim)

    def _fitness(self, population):
        # Evaluate the fitness of the population
        return np.array([self.fobj(ind) for ind in population])

    def _separation(self, index):
        # Compute separation from neighbors
        sep = np.zeros(self.dim)
        for j in range(self.population_size):
            if j != index:
                sep += (self.population[index] - self.population[j])
        return sep

    def _alignment(self):
        # Compute alignment of the population
        return np.mean(self.velocity, axis=0)

    def _cohesion(self, index):
        # Compute cohesion towards the center of mass of the population
        center = np.mean(self.population, axis=0)
        return center - self.population[index]

    def _attraction_to_food(self, index):
        # Compute attraction towards the food source
        return self.food_position - self.population[index]

    def _distraction_from_enemy(self, index):
        # Compute distraction away from the enemy
        return self.population[index] - self.enemy_position

    def run(self):
        """
        Execute the Dragonfly Algorithm.
        
        Returns:
        tuple: Best solution found and its fitness.
        """
        for iteration in range(self.max_iterations):
            # Evaluate fitness of the population
            fitness = self._fitness(self.population)
            
            # Update the best solution found so far
            min_fitness_idx = np.argmin(fitness)
            if fitness[min_fitness_idx] < self.best_fitness:
                self.best_fitness = fitness[min_fitness_idx]
                self.best_solution = self.population[min_fitness_idx].copy()

            # Update the position of each dragonfly
            for i in range(self.population_size):
                sep = self._separation(i) * self.s
                align = self._alignment() * self.a
                coh = self._cohesion(i) * self.c
                food_attract = self._attraction_to_food(i) * self.f
                enemy_distraction = self._distraction_from_enemy(i) * self.e

                # Update velocity and position
                self.velocity[i] = (self.w * self.velocity[i] + sep + align + coh + food_attract + enemy_distraction)
                self.population[i] = np.clip(self.population[i] + self.velocity[i], self.A, self.B)

        return self.best_solution, self.best_fitness


# Sample Objective Function (e.g., Sphere function)
def sphere_function(x):
    return np.sum(x**2)

# Example usage
if __name__ == "__main__":
    A = -10
    B = 10
    population_size = 30  # Number of dragonflies
    max_iterations = 100  # Number of iterations
    dim = 5  # Dimensionality of the problem

    da = DragonflyAlgorithm(A, B, population_size, max_iterations, dim, sphere_function)
    best_solution, best_fitness = da.run()

    # Output the result
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness: {best_fitness}")
