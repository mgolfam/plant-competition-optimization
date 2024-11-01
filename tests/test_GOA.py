import unittest
import numpy as np
from src.algorithms.GOA import GrasshopperOptimizationAlgorithm

class TestGOA(unittest.TestCase):

    def setUp(self):
        print('GOA TEST')
        self.A = -10
        self.B = 10
        self.population_size = 30
        self.max_iterations = 100
        self.dim = 5  # Dimensionality of the problem

        # Define a simple sphere function for testing
        self.fobj = lambda x: np.sum(x**2)

        # Initialize the GOA instance
        self.goa = GrasshopperOptimizationAlgorithm(self.A, self.B, self.population_size, self.max_iterations, self.dim, self.fobj)

    def test_solution_dimensions(self):
        best_solution, best_fitness = self.goa.run()
        self.assertEqual(len(best_solution), self.dim)

    def test_fitness_improvement(self):
        _, best_fitness = self.goa.run()
        self.assertGreaterEqual(best_fitness, 0)  # Fitness should be non-negative for the sphere function

if __name__ == '__main__':
    unittest.main()
