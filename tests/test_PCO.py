import unittest
import numpy as np
from src.PCO_main import PlantCompetitionOptimization  # Assuming you have a main class

class TestPCO(unittest.TestCase):

    def setUp(self):
        # Setup for the Plant Competition Optimization algorithm
        self.lb = 0
        self.ub = 10
        self.dim = 2
        self.pco = PlantCompetitionOptimization(self.lb, self.ub, num_plants=20, max_iterations=100, dim=self.dim)

    def test_solution_dimensions(self):
        # Test that the solution has the correct dimensions
        best_solution, best_fitness = self.pco.run()
        self.assertEqual(len(best_solution), self.dim)

    def test_non_negative_fitness(self):
        # Test that the returned fitness is non-negative
        best_solution, best_fitness = self.pco.run()
        self.assertGreaterEqual(best_fitness, 0)

    def test_improvement_over_iterations(self):
        # Test that the fitness improves over iterations
        initial_best_solution, initial_best_fitness = self.pco.run()
        subsequent_best_solution, subsequent_best_fitness = self.pco.run()
        self.assertLessEqual(subsequent_best_fitness, initial_best_fitness)

if __name__ == '__main__':
    unittest.main()
