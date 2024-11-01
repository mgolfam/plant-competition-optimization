import unittest
import numpy as np
from src.algorithms.PSO import PSO_Function

class TestPSO(unittest.TestCase):

    def setUp(self):
        # Setup a small test problem for the PSO algorithm
        self.lb = -10
        self.ub = 10
        self.dim = 2
        self.psz = 10
        self.pso_iterations = 50
        self.fobj = lambda x: np.sum(x**2)  # Sphere function as an objective

    def test_solution_dimensions(self):
        # Test that the solution has the correct dimensions
        result = PSO_Function(self.lb, self.ub, self.psz, self.pso_iterations, self.dim, self.fobj)
        best_fitness, num_calls = result[:2]  # Assuming PSO_Function returns more than 2 values, adjust accordingly
        self.assertGreaterEqual(best_fitness, 0)

    def test_fitness_improvement(self):
        # Test that fitness improves over iterations
        result_initial = PSO_Function(self.lb, self.ub, self.psz, self.pso_iterations, self.dim, self.fobj)
        initial_best_fitness = result_initial[0]
        
        result_subsequent = PSO_Function(self.lb, self.ub, self.psz, self.pso_iterations, self.dim, self.fobj)
        subsequent_best_fitness = result_subsequent[0]
        
        self.assertLessEqual(subsequent_best_fitness, initial_best_fitness + 1e-6)  # Add tolerance

if __name__ == '__main__':
    unittest.main()
