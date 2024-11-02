import numpy as np
import random
import matplotlib.pyplot as plt
np.random.seed(42)  # Set a fixed seed for reproducibility

class GeneticAlgorithm:
    def __init__(self, A, B, population_size, generations, dim, fobj, mutation_rate=0.01, crossover_rate=0.7):
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
        # print('self.population',self.population)
        self.best_individual = None
        self.best_fitness = float('inf')
        self.convergence = []  # List to store best fitness values over generations

    def _evaluate_population(self):
        fitness = np.array([self.fobj(ind) for ind in self.population])
        return fitness

    def _select_parents(self, fitness):
        probabilities = 1 / (fitness + 1e-8)  # Inverse proportional for minimization
        probabilities /= probabilities.sum()
        selected_indices = np.random.choice(np.arange(self.population_size), size=2, p=probabilities)
        return self.population[selected_indices[0]], self.population[selected_indices[1]]

    def _crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            crossover_point = np.random.randint(1, self.dim)
            offspring1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            offspring2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        else:
            offspring1, offspring2 = parent1.copy(), parent2.copy()
        return offspring1, offspring2

    def _mutate(self, offspring):
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
        for generation in range(self.generations):
            fitness = self._evaluate_population()
            
            # Update best individual
            min_fitness = np.min(fitness)
            min_index = np.argmin(fitness)
            if min_fitness < self.best_fitness:
                self.best_fitness = min_fitness
                self.best_individual = self.population[min_index].copy()
            
            # Print the best fitness and individual of the current generation
            print(f"Generation {generation + 1}: Best Fitness = {self.best_fitness}, Best Individual = {self.best_individual}")

            # Track best fitness for plotting
            self.convergence.append(self.best_fitness)

            # Create new population
            self.population = self._create_new_population(fitness)
            
        return self.best_individual, self.best_fitness

    def plot_convergence(self):
        plt.plot(self.convergence, 'b*-', linewidth=1, markeredgecolor='r', markersize=5)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title("Genetic Algorithm's Convergence")
        plt.show()

# 2D Schaffer Function
def schaffer_function(x):
    x1, x2 = x[0], x[1]
    numerator = np.sin(np.sqrt(x1**2 + x2**2))**2 - 0.5
    denominator = (1 + 0.001 * (x1**2 + x2**2))**2
    return 0.5 + numerator / denominator

# Example usage
if __name__ == "__main__":
    A = -100
    B = 100
    population_size = 30  # Number of individuals in the population
    generations = 100  # Number of generations (iterations)
    dim = 2  # Dimensionality of the problem (2D for the Schaffer function)

    ga = GeneticAlgorithm(A, B, population_size, generations, dim, schaffer_function)
    best_individual, best_fitness = ga.run()

    # Output the final result
    print(f"Final Best individual found: {best_individual}")
    print(f"Final Best fitness: {best_fitness}")

    # Plot the convergence
    ga.plot_convergence()
