import numpy as np
import random
import matplotlib.pyplot as plt

class GeneticAlgorithm:
    def __init__(self, A, B, population_size, generations, dim, fobj, mutation_rate=0.01, crossover_rate=0.7):
        """
        Initialize the Genetic Algorithm.
        
        Parameters:
        A (float): Lower bound of the search space.
        B (float): Upper bound of the search space.
        population_size (int): Number of individuals in the population.
        generations (int): Number of generations (iterations).
        dim (int): Dimensionality of the problem.
        fobj (function): Objective function to minimize.
        mutation_rate (float): Probability of mutation.
        crossover_rate (float): Probability of crossover.
        """
        self.A = A
        self.B = B
        self.population_size = population_size
        self.generations = generations
        self.dim = dim
        self.fobj = fobj
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Initializing population with random values between A and B
        self.population = A + (B - A) * np.random.rand(population_size, dim)
        self.best_individual = None
        self.best_fitness = float('inf')
        self.convergence = []  # List to store best fitness values over generations

    def _evaluate_population(self):
        # Evaluates the fitness of the population
        fitness = np.array([self.fobj(ind) for ind in self.population])
        return fitness

    def _select_parents(self, fitness):
        # Roulette wheel selection
        probabilities = 1 / (fitness + 1e-8)  # Inverse proportional for minimization
        probabilities /= probabilities.sum()
        selected_indices = np.random.choice(np.arange(self.population_size), size=2, p=probabilities)
        return self.population[selected_indices[0]], self.population[selected_indices[1]]

    def _crossover(self, parent1, parent2):
        # Single-point crossover
        if np.random.rand() < self.crossover_rate:
            crossover_point = np.random.randint(1, self.dim)
            offspring1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            offspring2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        else:
            offspring1, offspring2 = parent1.copy(), parent2.copy()
        return offspring1, offspring2

    def _mutate(self, offspring):
        # Random mutation with mutation rate
        for i in range(self.dim):
            if np.random.rand() < self.mutation_rate:
                offspring[i] = self.A + (self.B - self.A) * np.random.rand()
        return offspring

    def _create_new_population(self, fitness):
        new_population = []
        for _ in range(self.population_size // 2):
            parent1, parent2 = self._select_parents(fitness)
            offspring1, offspring2 = self._crossover(parent1, parent2)
            new_population.append(self._mutate(offspring1))
            new_population.append(self._mutate(offspring2))
        return np.array(new_population)

    def run(self):
        """
        Execute the Genetic Algorithm for a specified number of generations.
        
        Returns:
        tuple: Best individual found and its fitness.
        """
        for generation in range(self.generations):
            fitness = self._evaluate_population()
            
            # Update best individual
            min_fitness = np.min(fitness)
            min_index = np.argmin(fitness)
            if min_fitness < self.best_fitness:
                self.best_fitness = min_fitness
                self.best_individual = self.population[min_index].copy()
                
            # Track best fitness for plotting
            self.convergence.append(self.best_fitness)

            # Create new population
            self.population = self._create_new_population(fitness)
            
        return self.best_individual, self.best_fitness

    def plot_convergence(self):
        # Plot the convergence graph
        plt.plot(self.convergence, 'b*-', linewidth=1, markeredgecolor='r', markersize=5)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title("Genetic Algorithm's Convergence")
        plt.show()

# Sample Objective Function (e.g., Sphere function)
def sphere_function(x):
    return np.sum(x**2)

# Example usage
if __name__ == "__main__":
    A = -10
    B = 10
    population_size = 30  # Number of individuals in the population
    generations = 100  # Number of generations (iterations)
    dim = 5  # Dimensionality of the problem

    ga = GeneticAlgorithm(A, B, population_size, generations, dim, sphere_function)
    best_individual, best_fitness = ga.run()

    # Output the result
    print(f"Best individual found: {best_individual}")
    print(f"Best fitness: {best_fitness}")

    # Plot the convergence
    ga.plot_convergence()
