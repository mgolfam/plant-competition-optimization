import numpy as np
import matplotlib.pyplot as plt
from src.algorithms.GA import GeneticAlgorithm
from src.algorithms.SA import SimulatedAnnealing
from src.algorithms.PSO import ParticleSwarmOptimization

# Sample Objective Function (e.g., Sphere function)
def sphere_function(x):
    return np.sum(x**2)

# Parameters for the algorithms
A = -10
B = 10
dim = 5  # Dimensionality of the problem
generations = 100
population_size = 30
max_iterations = 100
initial_temperature = 1000
cooling_rate = 0.95

# Run Genetic Algorithm
ga = GeneticAlgorithm(A, B, population_size, generations, dim, sphere_function)
ga.run()

# Run Simulated Annealing
sa = SimulatedAnnealing(A, B, initial_temperature, cooling_rate, max_iterations, dim, sphere_function)
sa.run()

# Run Particle Swarm Optimization
pso = ParticleSwarmOptimization(A, B, population_size, max_iterations, dim, sphere_function)
pso.run()

# Plotting the convergence for all three algorithms
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) GA Convergence
axes[0].plot(ga.convergence, 'b*-', linewidth=1, markeredgecolor='r', markersize=5)
axes[0].set_title('GA Convergence')
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Best Fitness')

# (b) SA Convergence
axes[1].plot(sa.convergence, 'g*-', linewidth=1, markeredgecolor='r', markersize=5)
axes[1].set_title('SA Convergence')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Best Fitness')

# (c) PSO Convergence
axes[2].plot(pso.best_fitness, 'r*-', linewidth=1, markeredgecolor='b', markersize=5)
axes[2].set_title('PSO Convergence')
axes[2].set_xlabel('Iteration')
axes[2].set_ylabel('Best Fitness')

plt.tight_layout()
plt.show()
