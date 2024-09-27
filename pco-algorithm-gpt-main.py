import numpy as np

# Define the Schaffer function (an optimization problem)
def schaffer_function(x, y):
    numerator = np.sin(np.sqrt(x**2 + y**2))**2 - 0.5
    denominator = (1 + 0.001 * (x**2 + y**2))**2
    return 0.5 + numerator / denominator

# PCO Parameters
num_plants = 20
max_iterations = 200
max_size = 20
growth_rate = 0.1
seed_migration_rate = 0.05
search_space = (-100, 100)

# Initialize plants randomly
plants = np.random.uniform(search_space[0], search_space[1], (num_plants, 2))
fitness = np.array([schaffer_function(p[0], p[1]) for p in plants])

# Function to simulate PCO
def pco_optimization(num_iterations):
    global plants, fitness
    for iteration in range(num_iterations):
        new_plants = []
        
        # Growth and seed production
        for i, plant in enumerate(plants):
            # Create seeds around each plant (local search)
            num_seeds = int(max(1, fitness[i] * max_size))  # Seed count proportional to fitness
            for _ in range(num_seeds):
                seed = plant + np.random.uniform(-growth_rate, growth_rate, 2)  # Small local mutation
                new_plants.append(seed)
        
        # Add random migration seeds (global search)
        for _ in range(int(len(new_plants) * seed_migration_rate)):
            random_seed = np.random.uniform(search_space[0], search_space[1], 2)
            new_plants.append(random_seed)
        
        # Evaluate fitness of all plants and seeds
        all_plants = np.array(new_plants)
        all_fitness = np.array([schaffer_function(p[0], p[1]) for p in all_plants])
        
        # Select the best plants
        sorted_indices = np.argsort(all_fitness)
        plants = all_plants[sorted_indices[:num_plants]]
        fitness = all_fitness[sorted_indices[:num_plants]]

        # Print best fitness in each iteration
        best_fitness = np.min(fitness)
        print(f"Iteration {iteration+1}, Best Fitness: {best_fitness}")

# Run the PCO optimization for 50 iterations
pco_optimization(50)
