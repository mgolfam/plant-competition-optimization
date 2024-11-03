import numpy as np

class AntColonyOptimization:
    def __init__(self, distances, num_ants, num_iterations, alpha=1.0, beta=2.0, evaporation_rate=0.5):
        # Ensure the distance matrix is float type to handle np.inf
        self.distances = distances.astype(float)
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.num_cities = distances.shape[0]

    def _ant_solution(self):
        # Each ant starts from a random city
        solution = np.zeros((self.num_ants, self.num_cities), dtype=int)
        for i in range(self.num_ants):
            solution[i][0] = np.random.randint(self.num_cities)
        return solution

    def _update_pheromones(self, solutions, costs):
        self.pheromone *= self.evaporation_rate
        for solution, cost in zip(solutions, costs):
            for i in range(self.num_cities - 1):
                self.pheromone[solution[i], solution[i+1]] += 1.0 / cost

    def _calculate_probabilities(self, city, visited):
        pheromone = np.copy(self.pheromone[city])
        
        # Convert visited set to a list and use that for indexing
        visited_list = list(visited)
        pheromone[visited_list] = 0  # Set the pheromone of visited cities to 0
        
        distances = np.copy(self.distances[city])
        distances[visited_list] = np.inf  # Prevent ants from revisiting cities
        
        desirability = pheromone ** self.alpha * (1.0 / distances) ** self.beta
        return desirability / np.sum(desirability)

    def _generate_solution(self):
        solutions = np.zeros((self.num_ants, self.num_cities), dtype=int)
        for ant in range(self.num_ants):
            visited = set([solutions[ant][0]])
            for city in range(1, self.num_cities):
                probabilities = self._calculate_probabilities(solutions[ant][city-1], visited)
                next_city = np.random.choice(np.arange(self.num_cities), p=probabilities)
                solutions[ant][city] = next_city
                visited.add(next_city)
        return solutions

    def run(self):
        best_solution = None
        best_cost = float('inf')

        for iteration in range(self.num_iterations):
            solutions = self._generate_solution()
            costs = np.array([np.sum(self.distances[solution[:-1], solution[1:]]) for solution in solutions])

            self._update_pheromones(solutions, costs)

            min_cost = np.min(costs)
            if min_cost < best_cost:
                best_cost = min_cost
                best_solution = solutions[np.argmin(costs)]

            print(f"Iteration {iteration+1}/{self.num_iterations}, Best Cost: {best_cost}")

        return best_solution, best_cost

# Example: Distance matrix for 5 cities
if __name__ == "__main__":
    distances = np.array([
        [0, 2, 2, 5, 7],
        [2, 0, 4, 8, 2],
        [2, 4, 0, 1, 3],
        [5, 8, 1, 0, 6],
        [7, 2, 3, 6, 0]
    ], dtype=float)  # Ensure this matrix is of type float

    # Initialize ACO and run
    aco = AntColonyOptimization(distances, num_ants=10, num_iterations=100)
    best_solution, best_cost = aco.run()

    print(f"Best solution: {best_solution}")
    print(f"Best cost: {best_cost}")
