import numpy as np
import random
import matplotlib.pyplot as plt
np.random.seed(42)  # Set a fixed seed for reproducibility

class GeneticAlgorithm:
    def __init__(self, population_size, generations, dim, mutation_rate=0.01, crossover_rate=0.7):
        self.population_size = population_size
        self.generations = generations
        self.dim = dim
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Initialize population with random binary vectors
        self.population = np.random.randint(2, size=(population_size, dim))
        self.best_individual = None
        self.best_fitness = -float('inf')  # Maximization problem
        self.convergence = []  # List to store best fitness values over generations

    def _evaluate_population(self):
        # Fitness is the number of ones in the vector
        fitness = np.array([np.sum(ind) for ind in self.population])
        return fitness

    def _select_parents(self, fitness):
        probabilities = fitness / (fitness.sum() + 1e-8)  # Proportional for maximization
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
                offspring[i] = 1 - offspring[i]  # Flip the bit (0 becomes 1, 1 becomes 0)
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
            max_fitness = np.max(fitness)
            max_index = np.argmax(fitness)
            if max_fitness > self.best_fitness:
                self.best_fitness = max_fitness
                self.best_individual = self.population[max_index].copy()
            
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

# Example usage for "Max One" Problem
if __name__ == "__main__":
    population_size = 50  # Number of individuals in the population
    generations = 1000  # Number of generations (iterations)
    dim = 40  # Length of the binary vector

    ga = GeneticAlgorithm(population_size, generations, dim)
    best_individual, best_fitness = ga.run()

    # Output the final result
    print(f"Final Best individual found: {best_individual}")
    print(f"Final Best fitness: {best_fitness}")

    # Plot the convergence
    ga.plot_convergence()
