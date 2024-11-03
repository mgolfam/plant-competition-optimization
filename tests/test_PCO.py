import unittest
import numpy as np
from src.algorithms.PCO import PlantCompetitionOptimization
from src.problems_functions.benchmark_functions import schaffer_function  # Import the function object

class TestPCO(unittest.TestCase):

    def setUp(self):
        print('PCO TEST')
        # Setup for the Plant Competition Optimization algorithm
        self.lb = -5
        self.ub = 5
        self.dim = 2  # Ensure this matches the dimension of your problem
        self.pco = PlantCompetitionOptimization(
            fobj=schaffer_function,  # Use the function object directly
            n=20, 
            vmax=10, 
            Noi=100, 
            MaxPlantNumber=1000, 
            lb=self.lb, 
            ub=self.ub, 
            dim=self.dim
        )

    def test_solution_dimensions(self):
        # Test that the solution has the correct dimensions
        best_solution, best_fitness = self.pco.run()  # Expecting both solution and fitness
        self.assertEqual(len(best_solution), self.dim)  # Check the length of the solution

    def test_valid_fitness(self):
        # Test that the returned fitness is a valid float
        best_solution, best_fitness = self.pco.run()  # Expecting both solution and fitness
        self.assertIsInstance(best_fitness, float)  # Ensure it's a float

if __name__ == '__main__':
    unittest.main()
