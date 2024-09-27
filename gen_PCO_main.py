import numpy as np
import matplotlib.pyplot as plt

# Define the objective function details (Ackley function for F23)
def Get_Functions_details(Function_name):
    lb = np.array([-15, -15])  # Example lower bounds
    ub = np.array([30, 30])    # Example upper bounds
    dim = 2                     # Example dimension
    fobj = lambda x: -20 * np.exp(-0.2 * np.sqrt(0.5 * np.sum(x**2))) - np.exp(0.1 * np.sum(np.cos(2 * np.pi * x))) + 20 + np.exp(1)
    return lb, ub, dim, fobj

# Parameters and Initialization
Function_name = 'F23'
n = 20  # Initial number of plants
vmax = 10  # Maximum plant size
Noi = 200  # Number of iterations
MaxPlantNumber = 1000  # Maximum number of plants
maxPlant = n
lb, ub, dim, fobj = Get_Functions_details(Function_name)

rmax = np.max(ub - lb)  # Use max value of rmax as a scalar
maxTeta = np.exp(-1)
teta = maxTeta
k = 0.1
miu = 0.05

A = lb
B = ub

# Seeding, initialization
plants = A + (B - A) * np.random.rand(n, dim)
v = np.random.rand(n)
r = np.zeros(n)
plantNumber = n
iteration = 1
best = []
x = plants

migrantSeedsNo = 0
migrantPlant = []

# Main optimization loop
while plantNumber <= MaxPlantNumber and iteration <= Noi:
    # Fitness calculation
    f = np.array([fobj(plants[i, :]) for i in range(plantNumber)])
    
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
    survive = (fc >= sfc[maxPlant - 1])
    plants = plants[survive, :]  # Apply survival mask
    v = v[survive]  # Update plant sizes for survivors
    plantNumber = plants.shape[0]

    # Growth phase
    st = np.zeros(plantNumber)
    for i in range(plantNumber):
        r[i] = teta * rmax * np.exp(1 - (5 * v[i]) / vmax)  # Use rmax as scalar
        non = 0  # Count of neighbors
        for j in range(plantNumber):
            if np.sqrt(np.sum((plants[i, :] - plants[j, :])**2)) <= r[i]:
                st[i] += v[j]
                non += 1
        
        # Update plant size
        dv = fc[i] * k * (np.log(non * vmax) - np.log(st[i]))
        if v[i] + dv < vmax:
            v[i] += dv
        else:
            v[i] = vmax
    
    # SEED PRODUCTION
    sumNos = 0
    NOS = np.zeros(plantNumber, dtype=int)
    for i in range(plantNumber):
        NOS[i] = int(np.floor(v[i] + 1))
        sumNos += NOS[i]
        for j in range(NOS[i]):
            RND = np.random.randint(dim)
            Temp = (plants[i, RND] - r[i]) + 2 * r[i] * np.random.rand()
            seed = plants[i, :].copy()
            seed[RND] = Temp
            plants = np.vstack((plants, seed))
            v = np.append(v, np.random.rand())  # Add seed sizes

    # SEED MIGRATION
    migrantSeedsNoOld = migrantSeedsNo
    migrantSeedsNo = int(np.floor(miu * sumNos))
    migrantPlantOld = migrantPlant
    migrantPlant = np.random.randint(0, plantNumber + sumNos, migrantSeedsNo)
    
    for i in range(migrantSeedsNo):
        temp = A + (B - A) * np.random.rand(1, dim)
        plants[migrantPlant[i], :] = temp
    
    plantNumber = plants.shape[0]
    iteration += 1

# Plotting the convergence
plt.plot(best, 'b*-', linewidth=1, markeredgecolor='r', markersize=5)
plt.xlabel('Iteration')
plt.ylabel('Minimum Fitness Value')
plt.title("Algorithm's Convergence")
plt.show()

# Final Results
print(f'Minimum fitness achieved: {np.min(best)}')
