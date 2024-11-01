import unittest
import numpy as np
from src.algorithms.PSO import ParticleSwarmOptimization  # Adjust import to match your updated class

class TestParticleSwarmOptimization(unittest.TestCase):

    def setUp(self):
        # Setup a small test problem for the PSO algorithm
        self.lb = -10
        self.ub = 10
        self.dim = 2
        self.psz = 10
        self.pso_iterations = 50
        self.fobj = lambda x: np.sum(x**2)  # Sphere function as an objective

        # Initialize the PSO instance
        self.pso = ParticleSwarmOptimization(self.lb, self.ub, self.psz, self.pso_iterations, self.dim, self.fobj)

    def test_solution_dimensions(self):
        # Test that the solution has the correct dimensions
        best_position, best_fitness = self.pso.run()
        self.assertEqual(len(best_position), self.dim)
        self.assertGreaterEqual(best_fitness, 0)

    def test_fitness_improvement(self):
        # Test that fitness improves over iterations
        self.pso.run()  # Run PSO to populate best_fitness list
        initial_fitness = self.pso.best_fitness[0]
        final_fitness = self.pso.best_fitness[-1]
        
        # Check that the final fitness is not worse than the initial fitness
        self.assertLessEqual(final_fitness, initial_fitness)

if __name__ == '__main__':
    unittest.main()
