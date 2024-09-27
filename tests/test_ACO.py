import unittest
import numpy as np
from src.ACO_Function import AntColonyOptimization

class TestACO(unittest.TestCase):

    def setUp(self):
        # Setup a small test problem for the ACO algorithm
        self.distances = np.array([
            [0, 2, 2, 5, 7],
            [2, 0, 4, 8, 2],
            [2, 4, 0, 1, 3],
            [5, 8, 1, 0, 6],
            [7, 2, 3, 6, 0]
        ], dtype=float)

        self.aco = AntColonyOptimization(self.distances, num_ants=10, num_iterations=50)

    def test_solution_length(self):
        # Test that the solution length is correct
        best_solution, best_cost = self.aco.run()
        self.assertEqual(len(best_solution), len(self.distances))

    def test_non_negative_cost(self):
        # Test that the returned cost is non-negative
        best_solution, best_cost = self.aco.run()
        self.assertGreaterEqual(best_cost, 0)

    def test_improvement_over_iterations(self):
        # Test that the cost improves over iterations
        initial_best_solution, initial_best_cost = self.aco.run()
        subsequent_best_solution, subsequent_best_cost = self.aco.run()
        self.assertLessEqual(subsequent_best_cost, initial_best_cost)

if __name__ == '__main__':
    unittest.main()
