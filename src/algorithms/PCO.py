import numpy as np
import matplotlib.pyplot as plt
from src.problems_functions.benchmark_functions import schaffer_function

class PlantCompetitionOptimization:
    def __init__(self, fobj, n=20, vmax=10, Noi=200, MaxPlantNumber=1000, lb=-100, ub=100, dim=2):
        self.fobj = fobj  # Use the function object directly
        self.n = n
        self.vmax = vmax
        self.Noi = Noi
        self.MaxPlantNumber = MaxPlantNumber
        self.maxPlant = n
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.rmax = np.max(ub - lb)  # Use max value of rmax as a scalar
        self.maxTeta = np.exp(-1)
        self.teta = self.maxTeta
        self.k = 0.1
        self.miu = 0.05
        self.best_fitness = []

    def run(self):
        # Initialization
        np.random.seed(42)  # For reproducibility
        plants = self.lb + (self.ub - self.lb) * np.random.rand(self.n, self.dim)
        v = np.random.rand(self.n)
        r = np.zeros(self.n)
        plantNumber = self.n
        iteration = 1
        best = []
        migrantSeedsNo = 0
        migrantPlant = []

        # Main optimization loop
        while plantNumber <= self.MaxPlantNumber and iteration <= self.Noi:
            # Fitness calculation
            f = np.array([self.fobj(plants[i, :]) for i in range(plantNumber)])
            
            # Calculate fitness coefficients
            best.append(np.min(f))
            fn = f / np.linalg.norm(f)
            fitness = 1 / (1 + fn)
            mx = np.max(fitness)
            mn = np.min(fitness)
            if mn == mx:
                fc = fitness / mx
            else:
                fc = (fitness - mn) / (mx - mn)

            # Sort fitness and apply survival filtering
            sfc = np.sort(fc)[::-1]
            survive = (fc >= sfc[self.maxPlant - 1])
            plants = plants[survive, :]
            v = v[survive]
            plantNumber = plants.shape[0]

            # Growth phase
            st = np.zeros(plantNumber)
            for i in range(plantNumber):
                r[i] = self.teta * self.rmax * np.exp(1 - (5 * v[i]) / self.vmax)
                non = 0
                for j in range(plantNumber):
                    if np.sqrt(np.sum((plants[i, :] - plants[j, :])**2)) <= r[i]:
                        st[i] += v[j]
                        non += 1

                dv = fc[i] * self.k * (np.log(non * self.vmax) - np.log(st[i]))
                if v[i] + dv < self.vmax:
                    v[i] += dv
                else:
                    v[i] = self.vmax

            # SEED PRODUCTION
            sumNos = 0
            NOS = np.zeros(plantNumber, dtype=int)
            for i in range(plantNumber):
                NOS[i] = int(np.floor(v[i] + 1))
                sumNos += NOS[i]
                for j in range(NOS[i]):
                    RND = np.random.randint(self.dim)
                    Temp = (plants[i, RND] - r[i]) + 2 * r[i] * np.random.rand()
                    seed = plants[i, :].copy()
                    seed[RND] = Temp
                    plants = np.vstack((plants, seed))
                    v = np.append(v, np.random.rand())

            # SEED MIGRATION
            migrantSeedsNoOld = migrantSeedsNo
            migrantSeedsNo = int(np.floor(self.miu * sumNos))
            migrantPlantOld = migrantPlant
            migrantPlant = np.random.randint(0, plantNumber + sumNos, migrantSeedsNo)
            
            for i in range(migrantSeedsNo):
                temp = self.lb + (self.ub - self.lb) * np.random.rand(1, self.dim)
                plants[migrantPlant[i], :] = temp
            
            plantNumber = plants.shape[0]
            iteration += 1

        self.best_fitness = best
        best_solution = plants[np.argmin(f)]
        best_fitness = np.min(f)
        return best_solution, best_fitness

    def plot_convergence(self):
        plt.plot(self.best_fitness, 'b*-', linewidth=1, markeredgecolor='r', markersize=5)
        plt.xlabel('Iteration')
        plt.ylabel('Minimum Fitness Value')
        plt.title("PCO Algorithm's Convergence")
        plt.show()

# Example usage
if __name__ == '__main__':
    # Use the imported Ackley function
    pco = PlantCompetitionOptimization(fobj=schaffer_function, n=20, vmax=10, Noi=200, MaxPlantNumber=1000, lb=-5, ub=5, dim=2)
    best_solution, best_fitness = pco.run()
    print(f'Best solution: {best_solution}')
    print(f'Best fitness: {best_fitness}')
    pco.plot_convergence()
