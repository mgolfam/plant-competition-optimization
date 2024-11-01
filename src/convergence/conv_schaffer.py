import numpy as np
import matplotlib.pyplot as plt

from src.problems_functions.schaffer_function import schaffer_function_vec as schaffer_function
from src.algorithms.GA import GeneticAlgorithm
from src.algorithms.SA import SimulatedAnnealing
from src.algorithms.PSO import ParticleSwarmOptimization

# Example usage with the Schaffer function
if __name__ == "__main__":
    A = -10
    B = 10
    dim = 2  # The Schaffer function is typically used with 2 dimensions

    # Initialize and run Genetic Algorithm
    ga = GeneticAlgorithm(A, B, population_size=30, generations=100, dim=dim, fobj=schaffer_function)
    ga.run()

    # Initialize and run Simulated Annealing
    sa = SimulatedAnnealing(A, B, initial_temperature=1000, cooling_rate=0.95, max_iterations=500, dim=dim, fobj=schaffer_function)
    sa.run()

    # Initialize and run Particle Swarm Optimization
    pso = ParticleSwarmOptimization(A, B, psz=30, PsoIteration=100, dim=dim, fobj=schaffer_function)
    pso.run()

    # Plot the convergence for all three algorithms
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(ga.convergence, 'b*-', linewidth=1, markeredgecolor='r', markersize=5)
    axes[0].set_title('GA Convergence')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Best Fitness')

    axes[1].plot(sa.convergence, 'g*-', linewidth=1, markeredgecolor='r', markersize=5)
    axes[1].set_title('SA Convergence')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Best Fitness')

    axes[2].plot(pso.best_fitness, 'r*-', linewidth=1, markeredgecolor='b', markersize=5)
    axes[2].set_title('PSO Convergence')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Best Fitness')

    plt.tight_layout()
    plt.show()
