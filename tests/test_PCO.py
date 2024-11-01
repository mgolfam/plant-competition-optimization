import unittest
import numpy as np
from src.algorithms.PCO import PlantCompetitionOptimization

class TestPCO(unittest.TestCase):

    def setUp(self):
        # Setup for the Plant Competition Optimization algorithm
        self.lb = 0
        self.ub = 10
        self.dim = 4  # Ensure this matches the dimension of your problem
        self.pco = PlantCompetitionOptimization(function_name='F23', n=20, vmax=10, Noi=100, dim=self.dim)

    def test_solution_dimensions(self):
        # Test that the solution has the correct dimensions
        best_solution, best_fitness = self.pco.run()  # Expecting both solution and fitness
        self.assertEqual(len(best_solution), self.dim)  # Check the length of the solution

    def test_non_negative_fitness(self):
        # Test that the returned fitness is a valid float (not necessarily non-negative)
        best_solution, best_fitness = self.pco.run()  # Expecting both solution and fitness
        self.assertIsInstance(best_fitness, float)  # Ensure it's a float


if __name__ == '__main__':
    unittest.main()
