import numpy as np

class SalpSwarmAlgorithm:
    def __init__(self, fobj, lb, ub, dim, N, Max_iteration):
        self.fobj = fobj
        self.lb = lb if isinstance(lb, list) else [lb] * dim
        self.ub = ub if isinstance(ub, list) else [ub] * dim
        self.dim = dim
        self.N = N
        self.Max_iteration = Max_iteration

    def run(self):
        # Initialize positions and fitness
        SalpPositions = np.random.uniform(0, 1, (self.N, self.dim)) * (np.array(self.ub) - np.array(self.lb)) + np.array(self.lb)
        SalpFitness = np.full(self.N, float("inf"))

        FoodPosition = np.zeros(self.dim)
        FoodFitness = float("inf")

        for i in range(self.N):
            SalpFitness[i] = self.fobj(SalpPositions[i, :])

        sorted_salps_fitness = np.sort(SalpFitness)
        I = np.argsort(SalpFitness)
        Sorted_salps = np.copy(SalpPositions[I, :])

        FoodPosition = np.copy(Sorted_salps[0, :])
        FoodFitness = sorted_salps_fitness[0]

        for iteration in range(self.Max_iteration):
            c1 = 2 * np.exp(-((4 * iteration / self.Max_iteration) ** 2))

            for i in range(self.N):
                if i < self.N / 2:
                    for j in range(self.dim):
                        c2 = np.random.rand()
                        c3 = np.random.rand()

                        if c3 < 0.5:
                            SalpPositions[i, j] = FoodPosition[j] + c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])
                        else:
                            SalpPositions[i, j] = FoodPosition[j] - c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])
                else:
                    SalpPositions[i, :] = (SalpPositions[i-1, :] + SalpPositions[i, :]) / 2

            for i in range(self.N):
                SalpPositions[i, :] = np.clip(SalpPositions[i, :], self.lb, self.ub)
                SalpFitness[i] = self.fobj(SalpPositions[i, :])

                if SalpFitness[i] < FoodFitness:
                    FoodPosition = np.copy(SalpPositions[i, :])
                    FoodFitness = SalpFitness[i]

        return FoodPosition, FoodFitness

# Example objective function
def sphere_function(x):
    return np.sum(x ** 2)

# Example usage
if __name__ == "__main__":
    lb = -10
    ub = 10
    dim = 30
    N = 50
    Max_iteration = 100

    ssa = SalpSwarmAlgorithm(sphere_function, lb, ub, dim, N, Max_iteration)
    best_position, best_fitness = ssa.run()

    print(f"Best Position: {best_position}")
    print(f"Best Fitness: {best_fitness}")
