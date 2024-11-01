import unittest
import numpy as np
from src.algorithms.MPA import MarinePredatorsAlgorithm

class TestMarinePredatorsAlgorithm(unittest.TestCase):

    def setUp(self):
        print('MPA TEST')
        self.lb = -10
        self.ub = 10
        self.dim = 5
        self.population_size = 30
        self.max_iterations = 50
        self.fobj = lambda x: np.sum(x ** 2)  # Simple sphere function

        self.mpa = MarinePredatorsAlgorithm(self.lb, self.ub, self.dim, self.population_size, self.max_iterations, self.fobj)

    def test_solution_dimensions(self):
        best_position, _, _ = self.mpa.run()
        self.assertEqual(len(best_position), self.dim)

    def test_fitness_non_negative(self):
        _, best_fitness, _ = self.mpa.run()
        self.assertGreaterEqual(best_fitness, 0)

if __name__ == "__main__":
    unittest.main()
