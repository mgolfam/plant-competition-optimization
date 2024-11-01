import numpy as np
import unittest

from src.algorithms.SSA import SalpSwarmAlgorithm  # Correct import path

class TestSSA(unittest.TestCase):
    def setUp(self):
        self.lb = -10
        self.ub = 10
        self.dim = 5
        self.N = 30
        self.Max_iteration = 100
        self.fobj = lambda x: np.sum(x ** 2)  # Simple sphere function

        self.ssa = SalpSwarmAlgorithm(self.fobj, self.lb, self.ub, self.dim, self.N, self.Max_iteration)

    def test_solution_dimensions(self):
        best_position, _ = self.ssa.run()
        self.assertEqual(len(best_position), self.dim)

    def test_fitness_non_negative(self):
        _, best_fitness = self.ssa.run()
        self.assertGreaterEqual(best_fitness, 0)

if __name__ == "__main__":
    unittest.main()
