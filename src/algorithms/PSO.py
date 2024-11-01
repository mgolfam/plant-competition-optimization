import numpy as np
import matplotlib.pyplot as plt

class ParticleSwarmOptimization:
    def __init__(self, A, B, psz, PsoIteration, dim, fobj):
        # Initialize PSO parameters
        self.A = A  # Lower bound
        self.B = B  # Upper bound
        self.psz = psz  # Number of particles
        self.PsoIteration = PsoIteration  # Number of iterations
        self.dim = dim  # Dimensionality
        self.fobj = fobj  # Objective function
        self.c1 = 1.0  # Cognitive coefficient
        self.c2 = 0.3  # Social coefficient
        self.inertia = 0.8  # Inertia weight
        self.best_fitness = []  # List to store the best fitness values over iterations
        self.PSO_Call = 0  # Number of function calls

    def run(self):
        # Step 1: Initialization
        np.random.seed(42)  # For reproducibility
        pso = self.A + (self.B - self.A) * np.random.rand(self.psz, self.dim)  # Particle positions
        vp = np.random.uniform(-1, 1, (self.psz, self.dim))  # Particle velocities

        # Evaluate initial fitness
        fitness = np.array([self.fobj(pso[i]) for i in range(self.psz)])
        self.PSO_Call += self.psz

        # Initialize Gbest and Pbest
        Gbest_index = np.argmin(fitness)
        Gbest_position = pso[Gbest_index].copy()
        Gbest_fitness = fitness[Gbest_index]
        Pbest = pso.copy()
        Pbest_fitness = fitness.copy()

        # Store the initial best fitness
        self.best_fitness.append(Gbest_fitness)

        # Main PSO loop
        for iteration in range(self.PsoIteration):
            # Step 2.1: Update velocities and positions
            r1, r2 = np.random.rand(2, self.psz, self.dim)
            vp = (self.inertia * vp +
                  self.c1 * r1 * (Pbest - pso) +
                  self.c2 * r2 * (Gbest_position - pso))
            pso = np.clip(pso + vp, self.A, self.B)  # Update positions and ensure within bounds

            # Step 2.2: Evaluate fitness for updated positions
            fitness = np.array([self.fobj(pso[i]) for i in range(self.psz)])
            self.PSO_Call += self.psz

            # Step 2.3: Update Pbest
            update_mask = Pbest_fitness > fitness
            Pbest[update_mask] = pso[update_mask]
            Pbest_fitness[update_mask] = fitness[update_mask]

            # Step 2.4: Update Gbest
            Gbest_index = np.argmin(Pbest_fitness)
            Gbest_position = Pbest[Gbest_index].copy()
            Gbest_fitness = Pbest_fitness[Gbest_index]

            # Store the best fitness for plotting
            self.best_fitness.append(Gbest_fitness)

        # Return the best fitness and the number of function calls
        return Gbest_position, Gbest_fitness

    def plot_convergence(self):
        # Plot the convergence graph
        plt.plot(self.best_fitness, 'b*-', linewidth=1, markeredgecolor='r', markersize=5)
        plt.xlabel('Iteration')
        plt.ylabel('Minimum Fitness')
        plt.title("PSO Algorithm's Convergence")
        plt.show()

# Example usage
if __name__ == "__main__":
    # Sample Objective Function (e.g., Sphere function)
    def sphere_function(x):
        return np.sum(x**2)

    # Initialize PSO with parameters
    A = -10
    B = 10
    psz = 30  # Number of particles
    PsoIteration = 100  # Number of iterations
    dim = 2  # Dimensionality of the problem

    # Create PSO instance and run
    pso = ParticleSwarmOptimization(A, B, psz, PsoIteration, dim, sphere_function)
    best_position, best_fitness = pso.run()
    print(f"Best fitness found: {best_fitness}")
    print(f"Best position: {best_position}")

    # Plot the convergence
    pso.plot_convergence()
