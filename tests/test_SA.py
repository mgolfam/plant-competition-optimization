import unittest
import numpy as np
from src.algorithms.SA import SimulatedAnnealing

class TestSimulatedAnnealing(unittest.TestCase):

    def setUp(self):
        print('SA TEST')
        self.A = -10
        self.B = 10
        self.initial_temperature = 1000
        self.cooling_rate = 0.95
        self.max_iterations = 100
        self.dim = 5  # Dimensionality of the problem

        # Define a simple sphere function for testing
        self.fobj = lambda x: np.sum(x**2)

        # Initialize the SA instance
        self.sa = SimulatedAnnealing(self.A, self.B, self.initial_temperature, self.cooling_rate, self.max_iterations, self.dim, self.fobj)

    def test_solution_dimensions(self):
        best_solution, best_fitness = self.sa.run()
        self.assertEqual(len(best_solution), self.dim)

    def test_fitness_improvement(self):
        _, best_fitness = self.sa.run()
        self.assertGreaterEqual(best_fitness, 0)  # Fitness should be non-negative for the sphere function

if __name__ == '__main__':
    unittest.main()
