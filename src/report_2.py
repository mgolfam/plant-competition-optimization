import inspect
import time
import csv
from src.problems_functions import functions_details  # Your benchmark functions module
from src.tools.csv_manager import CSVManager
from src.algorithms import PCO, GA, PSO, SA, GOA, DA, SSA, WOA, MPA

def generate_table_5():
    # Initialize CSV Manager to save results
    csv_manager = CSVManager("data/table_5_results.csv")
    headers = [
        "Function Name", "Definition", "Search Domain", "Number of Local Minima",
        "Global Minimum", "Algorithm", "Avg Result", "Execution Time (s)", "Error Message"
    ]
    csv_manager.write_headers(headers)

    # Load functions from functions_details
    functions = inspect.getmembers(functions_details, inspect.isfunction)

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

    # Iterate over each function and algorithm
    for func_name, func in functions:
        for algo_name, AlgoClass in algorithms.items():
            print(f"Running {algo_name} for {func_name}...")
            try:
                start_time = time.time()
                algo_instance = AlgoClass(func)
                best_solution, best_fitness = algo_instance.run()
                execution_time = time.time() - start_time
                print(f"{func_name} ({algo_name}): Best solution: {best_solution}, Avg Result: {best_fitness}")

                # Fetch function details from the document
                function_details = {
                    "Function Name": func_name,
                    "Definition": "Your function definition here",
                    "Search Domain": "Your search domain",
                    "Number of Local Minima": "Number of minima",
                    "Global Minimum": "Global minimum",
                    "Algorithm": algo_name,
                    "Avg Result": best_fitness,
                    "Execution Time (s)": execution_time,
                    "Error Message": ""
                }

                csv_manager.append_row(function_details)
            except Exception as e:
                print(f"Error running {algo_name} on {func_name}: {e}")
                csv_manager.append_row({
                    "Function Name": func_name,
                    "Definition": "Your function definition here",
                    "Search Domain": "Your search domain",
                    "Number of Local Minima": "Number of minima",
                    "Global Minimum": "Global minimum",
                    "Algorithm": algo_name,
                    "Avg Result": "N/A",
                    "Execution Time (s)": "N/A",
                    "Error Message": str(e)
                })

if __name__ == "__main__":
    generate_table_5()
