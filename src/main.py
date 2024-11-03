import inspect
from src.problems_functions import benchmark_functions  # Import your benchmark functions module
from src.algorithms.PCO import PlantCompetitionOptimization  # Import your PCO class

def main():
    # Retrieve all functions from the benchmark_functions module
    functions = inspect.getmembers(benchmark_functions, inspect.isfunction)

    # Loop over each function and run the PCO algorithm
    for name, func in functions:
        print(f"Running PCO for {name} function...")
        pco = PlantCompetitionOptimization(fobj=func, n=20, vmax=10, Noi=200, MaxPlantNumber=1000, lb=-5, ub=5, dim=2)
        try:
            best_solution, best_fitness = pco.run()
            print(f"{name} function: Best solution: {best_solution}, Best fitness: {best_fitness}")
        except Exception as e: print(e)
        # pco.plot_convergence()

if __name__ == '__main__':
    main()
