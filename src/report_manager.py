from src.problems_functions import benchmark_functions  # Your benchmark functions
from src.tools.csv_manager import CSVManager  # Using CSVManager for handling CSV operations
from src.algorithms import (
    PCO, GA, PSO, SA, GOA, DA, SSA, WOA, MPA  # Import all algorithms
)
import inspect
import time

def run_all_algorithms():
    csv_manager = CSVManager("data/full_algorithm_comparison.csv")
    headers = [
        "Function Name", "Algorithm", "Best Solution", "Best Fitness",
        "Execution Time (s)", "Error Message"
    ]
    csv_manager.write_headers(headers)

    functions = inspect.getmembers(benchmark_functions, inspect.isfunction)

    algorithms = {
        "PCO": lambda func: PCO.PlantCompetitionOptimization(
            fobj=func, n=20, vmax=10, Noi=200, MaxPlantNumber=1000, lb=-5, ub=5, dim=2
        ),
        "GA": lambda func: GA.GeneticAlgorithm(
            fobj=func, A=10, B=0.5, dim=2, population_size=50, generations=200
        ),
        "PSO": lambda func: PSO.ParticleSwarmOptimization(
            fobj=func, psz=30, PsoIteration=200, A=-5, B=5, dim=2
        ),
        "SA": lambda func: SA.SimulatedAnnealing(
            fobj=func, A=10, B=0.5, initial_temperature=1000, cooling_rate=0.9, max_iterations=200, dim=2
        ),
        "GOA": lambda func: GOA.GrasshopperOptimizationAlgorithm(
            fobj=func, population_size=30, max_iterations=200, A=-5, B=5, dim=2
        ),
        "DA": lambda func: DA.DragonflyAlgorithm(
            fobj=func, population_size=30, max_iterations=200, A=-5, B=5, dim=2
        ),
        "SSA": lambda func: SSA.SalpSwarmAlgorithm(
            fobj=func, N=30, Max_iteration=200, lb=-5, ub=5, dim=2
        ),
        "WOA": lambda func: WOA.WhaleOptimizationAlgorithm(
            fobj=func, population_size=30, max_iterations=200, lb=-5, ub=5, dim=2
        ),
        "MPA": lambda func: MPA.MarinePredatorsAlgorithm(
            fobj=func, population_size=30, max_iterations=200, lb=-5, ub=5, dim=2
        )
    }

    for func_name, func in functions:
        for algo_name, AlgoClass in algorithms.items():
            print(f"Running {algo_name} for {func_name} function...")
            try:
                start_time = time.time()
                algo_instance = AlgoClass(func)
                best_solution, best_fitness = algo_instance.run()
                execution_time = time.time() - start_time
                print(f"{func_name} ({algo_name}): Best solution: {best_solution}, Best fitness: {best_fitness}")

                csv_manager.append_row({
                    "Function Name": func_name,
                    "Algorithm": algo_name,
                    "Best Solution": str(best_solution),
                    "Best Fitness": best_fitness,
                    "Execution Time (s)": execution_time,
                    "Error Message": ""
                })
            except Exception as e:
                print(f"Error running {algo_name} on {func_name}: {e}")
                csv_manager.append_row({
                    "Function Name": func_name,
                    "Algorithm": algo_name,
                    "Best Solution": "N/A",
                    "Best Fitness": "N/A",
                    "Execution Time (s)": "N/A",
                    "Error Message": str(e)
                })

if __name__ == "__main__":
    run_all_algorithms()
