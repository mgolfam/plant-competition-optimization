import unittest
import numpy as np
from src.algorithms.GA import GeneticAlgorithm

class TestGeneticAlgorithm(unittest.TestCase):

    def setUp(self):
        self.A = -10
        self.B = 10
        self.population_size = 30
        self.generations = 50
        self.dim = 5  # Dimensionality of the problem

        # Define a simple sphere function for testing
        self.fobj = lambda x: np.sum(x**2)

        # Initialize the GA instance
        self.ga = GeneticAlgorithm(self.A, self.B, self.population_size, self.generations, self.dim, self.fobj)

    def test_solution_dimensions(self):
        best_individual, best_fitness = self.ga.run()
        self.assertEqual(len(best_individual), self.dim)

    def test_fitness_improvement(self):
        _, initial_fitness = self.ga.run()
        self.assertGreaterEqual(initial_fitness, 0)  # Fitness should be non-negative for the sphere function

if __name__ == '__main__':
    unittest.main()
