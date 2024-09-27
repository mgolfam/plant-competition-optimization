import numpy as np
import matplotlib.pyplot as plt

# Define the objective function details (Ackley function for F23)
def get_function_details(Function_name):
    if Function_name == 'F23':
        lb = -32.768
        ub = 32.768
        dim = 2
        
        def fobj(x):
            return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0]**2 + x[1]**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))) + 20 + np.e
        
        return lb, ub, dim, fobj

# Initialize parameters
Function_name = 'F23'
print(f'Function_name = {Function_name}')
n = 20
print(f'Number of initial plants = {n}')
vmax = 10
Noi = 200
print(f'maximum number of algorithms iteration = {Noi}')
MaxPlantNumber = 1000
print(f'Maximum of Plants Number = {MaxPlantNumber}')

# Get function details
lb, ub, dim, fobj = get_function_details(Function_name)
rmax = ub - lb
maxTeta = np.exp(-1)
print(f'max Teta = {maxTeta}')
teta = maxTeta
k = 0.1
miu = 0.05
A = lb
B = ub

# Seeding (initialization)
plants = A + (B - A) * np.random.rand(n, dim)
r = np.zeros(n)
v = np.random.rand(n)
plantNumber = n
iteration = 1
best = []
migrantSeedsNo = 0
migrantPlant = []

# Main optimization loop
while plantNumber <= MaxPlantNumber and iteration <= Noi:
    
    # Calculate fitness
    f = np.array([fobj(plants[i, :]) for i in range(plantNumber)])
    best.append(np.min(f))

    fn = f / np.linalg.norm(f)
    fitness = 1.0 / (1 + fn)

    mx = np.max(fitness)
    mn = np.min(fitness)

    if mn == mx:
        fc = fitness / mx
    else:
        fc = (fitness - mn) / (mx - mn)

    # Sort fitness coefficients and determine which plants survive
    sfc = np.sort(fc)[::-1]  # Sort in descending order
    max_survivors = min(len(sfc), n)  # Ensure we only keep the best `n` plants
    survive = fc >= sfc[max_survivors - 1]  # The `n` here corresponds to the initial number of plants

    # Apply survival filtering BEFORE adding new seeds
    plants = plants[survive, :]
    v = v[survive]  # Filter corresponding plant sizes
    plantNumber = plants.shape[0]  # Update plant number

    # Growth and seeding
    st = np.zeros(plantNumber)
    for i in range(plantNumber):
        r[i] = teta * rmax * np.exp(1 - (5 * v[i]) / vmax)
        non = 0
        for j in range(plantNumber):
            if np.linalg.norm(plants[i, :] - plants[j, :]) <= r[i]:
                st[i] += v[j]
                non += 1
        dv = fc[i] * k * (np.log(non * vmax) - np.log(st[i]))
        v[i] = min(v[i] + dv, vmax)
    
    # Seed production AFTER filtering
    NOS = np.floor(v + 1).astype(int)
    sumNos = np.sum(NOS)
    for i in range(plantNumber):
        for j in range(NOS[i]):
            RND = np.random.randint(dim)
            temp = plants[i, RND] - r[i] + 2 * r[i] * np.random.rand()
            seed = plants[i, :].copy()
            seed[RND] = temp
            plants = np.vstack([plants, seed])
            v = np.append(v, np.random.rand())  # Update plant size array

    # Seed migration AFTER filtering and seeding
    migrantSeedsNoOld = migrantSeedsNo
    migrantSeedsNo = int(miu * sumNos)
    migrantPlantOld = migrantPlant

    # Ensure plants array has enough space before seed migration
    plants = np.vstack([plants, np.zeros((migrantSeedsNo, dim))])  # Add space for migrant seeds
    v = np.append(v, np.random.rand(migrantSeedsNo))  # Update plant size array for new migrant seeds

    # Assign new random positions to migrant plants
    for i in range(migrantSeedsNo):
        plants[plantNumber + i, :] = A + (B - A) * np.random.rand(dim)  # Assign to the new rows

    plantNumber += migrantSeedsNo  # Update plantNumber to reflect newly added plants
    iteration += 1

# Plot results
plt.plot(best, 'b*-', linewidth=1, markeredgecolor='r', markersize=5)
plt.xlabel('Iteration')
plt.ylabel('Minimum')
plt.title("Algorithm's Convergence")
plt.show()

# Final results
print(f'Minimum fitness: {np.min(best)}')
print(f'Maximum fitness: {np.max(f)}')
print(f'Maximum plant size: {np.max(v)}')
print(f'Maximum neighborhood radius: {np.max(r)}')
