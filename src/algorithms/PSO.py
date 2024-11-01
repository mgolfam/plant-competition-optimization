import numpy as np
import matplotlib.pyplot as plt

def PSO_Function(A, B, psz, PsoIteration, dim, fobj):
    """
    Particle Swarm Optimization (PSO) Algorithm.

    Parameters:
    A (float): Lower bound of the search space.
    B (float): Upper bound of the search space.
    psz (int): Number of particles.
    PsoIteration (int): Number of iterations.
    dim (int): Dimensionality of the problem.
    fobj (function): Objective function to minimize.

    Returns:
    tuple: Best fitness found, number of function calls, list of best fitness values at each iteration.
    """
    # PSO parameters
    c1 = 1.0  # Cognitive coefficient
    c2 = 0.3  # Social coefficient
    inertia = 0.8  # Inertia weight

    # Step 1: Initialization
    pso = A + (B - A) * np.random.rand(psz, dim)  # Initial positions of particles
    vp = np.random.normal(0, 1, (psz, dim))  # Initial velocities of particles
    PSO_Call = 0

    # Evaluate fitness for each particle
    f = np.array([fobj(pso[i]) for i in range(psz)])
    PSO_Call += psz

    # Find the particle with the best fitness
    mn = np.min(f)
    index1 = np.argmin(f)
    bestPso = [mn]
    Best_particle = pso[index1]
    fitness = f

    # Initialize Gbest (Global Best)
    Gbest = np.argmin(fitness)

    # Initialize Pbest (Personal Best)
    Pbest = pso.copy()
    Pbest_fitness = fitness.copy()

    # Main PSO loop
    for iteration in range(PsoIteration):
        # Step 2.1: Find Gbest (best particle in the swarm)
        Gbest = np.argmin(fitness)
        Best_Fitness = fitness[Gbest]
        bestPso.append(Best_Fitness)
        Best_particle = pso[Gbest]

        # Step 2.2: Update Pbest (personal best positions)
        update_mask = Pbest_fitness > fitness
        Pbest[update_mask] = pso[update_mask]
        Pbest_fitness[update_mask] = fitness[update_mask]

        # Step 2.3: Update velocities and positions of particles
        r1, r2 = np.random.rand(2, psz, dim)
        vp = (inertia * vp +
              c1 * r1 * (Pbest - pso) +
              c2 * r2 * (pso[Gbest] - pso))
        pso += vp

        # Step 2.4: Evaluate fitness for updated positions
        f = np.array([fobj(pso[i]) for i in range(psz)])
        fitness = f
        PSO_Call += psz

    # Find the best result after all iterations
    PSO_Best = np.min(bestPso)
    return PSO_Best, PSO_Call, bestPso

def main():
    # Sample Objective Function (e.g., Sphere function)
    def sphere_function(x):
        return np.sum(x**2)

    # Example usage of PSO_Function
    A = -10
    B = 10
    psz = 30  # Number of particles
    PsoIteration = 100  # Number of iterations
    dim = 2  # Dimensionality of the problem

    # Call the PSO function
    PSO_Best, PSO_Call, bestPso = PSO_Function(A, B, psz, PsoIteration, dim, sphere_function)

    # Output the result
    print(f"Best fitness found: {PSO_Best}")
    print(f"Number of function calls: {PSO_Call}")

    # Plotting the convergence
    plt.plot(bestPso, 'b*-', linewidth=1, markeredgecolor='r', markersize=5)
    plt.xlabel('Iteration')
    plt.ylabel('Minimum Fitness')
    plt.title('Convergence of PSO Algorithm')
    plt.show()

if __name__ == "__main__":
    main()
