import numpy as np

def PSO_Function(A, B, psz, PsoIteration, dim, fobj):
    # PSO parameters
    c1 = 1.0
    c2 = 0.3
    inertia = 0.8

    # Step 1: Initialization
    pso = A + (B - A) * np.random.rand(psz, dim)
    Best_particle = []
    bestPso = []
    vp = np.random.normal(0, 1, (psz, dim))
    PSO_Call = 0

    # Evaluate fitness
    f = np.array([fobj(pso[i]) for i in range(psz)])
    PSO_Call += psz

    mn = np.min(f)
    index1 = np.argmin(f)
    bestPso.append(mn)
    Best_particle.append(pso[index1])
    fitness = f

    # Initialize Gbest
    Gbest = np.argmin(fitness)

    # Initialize Pbest
    Pbest = pso.copy()
    Pbest_fitness = f.copy()
    bestPso = []

    # Main PSO loop
    for iteration in range(PsoIteration):
        # Step 2.1: Find Gbest
        Gbest = np.argmin(fitness)
        Best_Fitness = fitness[Gbest]
        bestPso.append(Best_Fitness)
        Best_particle = pso[Gbest]

        # Step 2.2: Update Pbest
        update_mask = Pbest_fitness > fitness
        Pbest[update_mask] = pso[update_mask]
        Pbest_fitness[update_mask] = fitness[update_mask]

        # Step 2.3: Update velocities and positions
        r1, r2 = np.random.rand(2, psz, dim)
        vp = (inertia * vp + 
              c1 * r1 * (Pbest - pso) + 
              c2 * r2 * (pso[Gbest] - pso))
        pso += vp

        # Step 2.4: Evaluate fitness
        f = np.array([fobj(pso[i]) for i in range(psz)])
        fitness = f
        PSO_Call += psz

    # Find best result
    Result = np.min(bestPso)
    PSO_Best = Result

    return PSO_Best, PSO_Call

