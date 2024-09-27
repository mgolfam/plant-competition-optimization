import numpy as np
import matplotlib.pyplot as plt
from functions_details import Get_Functions_details  # Import the details function from your file

class PlantCompetitionOptimization:
    def __init__(self, function_name='F23', n=20, vmax=10, Noi=200, MaxPlantNumber=1000, dim=None):
        # Set parameters
        self.Function_name = function_name
        self.n = n  # Initial number of plants
        self.vmax = vmax  # Maximum plant size
        self.Noi = Noi  # Number of iterations
        self.MaxPlantNumber = MaxPlantNumber  # Maximum number of plants
        self.maxPlant = n
        self.lb, self.ub, self.dim, self.fobj = Get_Functions_details(self.Function_name)
        self.rmax = np.max(self.ub - self.lb)  # Use max value of rmax as a scalar
        self.maxTeta = np.exp(-1)
        self.teta = self.maxTeta
        self.k = 0.1
        self.miu = 0.05
        self.A = self.lb
        self.B = self.ub
        self.best_fitness = []
        
        if dim:
            self.dim = dim

    def run(self):
        # Initialization
        np.random.seed(42)  # For reproducibility
        plants = self.A + (self.B - self.A) * np.random.rand(self.n, self.dim)
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
            fn = f / np.linalg.norm(f)  # Make sure norm is calculated in the same way as MATLAB
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
            plants = plants[survive, :]  # Apply survival mask
            v = v[survive]  # Update plant sizes for survivors
            plantNumber = plants.shape[0]

            # Growth phase
            st = np.zeros(plantNumber)
            for i in range(plantNumber):
                r[i] = self.teta * self.rmax * np.exp(1 - (5 * v[i]) / self.vmax)  # Use rmax as scalar
                non = 0  # Count of neighbors
                for j in range(plantNumber):
                    if np.sqrt(np.sum((plants[i, :] - plants[j, :])**2)) <= r[i]:
                        st[i] += v[j]
                        non += 1
                
                # Update plant size
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
                    v = np.append(v, np.random.rand())  # Add seed sizes

            # SEED MIGRATION
            migrantSeedsNoOld = migrantSeedsNo
            migrantSeedsNo = int(np.floor(self.miu * sumNos))
            migrantPlantOld = migrantPlant
            migrantPlant = np.random.randint(0, plantNumber + sumNos, migrantSeedsNo)
            
            for i in range(migrantSeedsNo):
                temp = self.A + (self.B - self.A) * np.random.rand(1, self.dim)
                plants[migrantPlant[i], :] = temp
            
            plantNumber = plants.shape[0]
            iteration += 1

        self.best_fitness = best
        return np.min(best)

    def plot_convergence(self):
        plt.plot(self.best_fitness, 'b*-', linewidth=1, markeredgecolor='r', markersize=5)
        plt.xlabel('Iteration')
        plt.ylabel('Minimum Fitness Value')
        plt.title("Algorithm's Convergence")
        plt.show()

# Example usage:
if __name__ == '__main__':
    pco = PlantCompetitionOptimization(function_name='F23', n=20, vmax=10, Noi=200, MaxPlantNumber=1000)
    min_fitness = pco.run()
    print(f'Minimum fitness achieved: {min_fitness}')
    pco.plot_convergence()
