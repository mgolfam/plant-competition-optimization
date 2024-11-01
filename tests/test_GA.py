import unittest
import numpy as np
from src.algorithms.GA import GeneticAlgorithm

class TestGeneticAlgorithm(unittest.TestCase):

    def setUp(self):
        print('GA TEST')
        # Setup parameters for the Genetic Algorithm
        self.A = -10
        self.B = 10
        self.population_size = 30
        self.generations = 50
        self.dim = 5  # Dimensionality of the problem

        # Define a simple sphere function as the objective function
        self.fobj = lambda x: np.sum(x**2)

        # Initialize the GA instance and run it
        self.ga = GeneticAlgorithm(self.A, self.B, self.population_size, self.generations, self.dim, self.fobj)
        self.ga.run()  # Run the algorithm to populate convergence data

    def test_solution_dimensions(self):
        # Test the solution's dimensionality
        best_individual, best_fitness = self.ga.best_individual, self.ga.best_fitness
        self.assertEqual(len(best_individual), self.dim, "Best individual does not have the correct dimensions.")

    def test_fitness_non_negative(self):
        # Check if the best fitness is non-negative
        self.assertGreaterEqual(self.ga.best_fitness, 0, "Best fitness should be non-negative for the sphere function.")

    def test_fitness_improvement_over_generations(self):
        # Check if the best fitness value improves over generations
        initial_fitness = self.ga.convergence[0]
        final_fitness = self.ga.convergence[-1]
        self.assertLessEqual(final_fitness, initial_fitness, "Final fitness should be less than or equal to initial fitness.")

if __name__ == '__main__':
    unittest.main()
