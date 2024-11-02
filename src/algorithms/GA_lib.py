import pygad
import numpy as np
import matplotlib.pyplot as plt
import random

np.random.seed(42)  # Set a fixed seed for NumPy
random.seed(42)  # Set a fixed seed for the built-in random module

# Define the fitness function
def fitness_func(ga_instance, solution, solution_idx):
    return -np.sum(solution**2)  # Negative sign for minimization

# Parameters
A, B = -10, 10
population_size, generations, dim = 30, 100, 5
mutation_rate, crossover_rate = 0.01, 0.7
gene_space = [{'low': A, 'high': B} for _ in range(dim)]

# Set up Pygad
ga_instance = pygad.GA(
    num_generations=generations,
    num_parents_mating=int(population_size * 0.5),
    fitness_func=fitness_func,
    sol_per_pop=population_size,
    num_genes=dim,
    gene_space=gene_space,
    parent_selection_type="rws",
    crossover_type="single_point",
    crossover_probability=crossover_rate,
    mutation_type="random",
    mutation_probability=mutation_rate,
)

ga_instance.run()

# Retrieve and print the best solution
solution, solution_fitness, _ = ga_instance.best_solution()
print(f"Best individual found: {solution}")
print(f"Best fitness: {solution_fitness}")

# Print convergence values
fitness_values = ga_instance.best_solutions_fitness
print("Convergence values (Best fitness over generations):")
for generation, fitness in enumerate(fitness_values):
    print(f"Generation {generation + 1}: Best Fitness = {fitness}")

# Plot the fitness over generations
plt.plot(fitness_values, 'b*-', linewidth=1, markeredgecolor='r', markersize=5)
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title("Genetic Algorithm's Convergence (Pygad)")
plt.show()
