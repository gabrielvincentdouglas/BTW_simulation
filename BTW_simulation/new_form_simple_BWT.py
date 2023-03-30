import numpy as np
import time
import scipy.sparse as sp

# Parameters
n = 50
size = n**2
threshold = 4
nsteps = 2000


def create_exchange(threshold):
    exchange = threshold*np.identity(size)
    exchange -= np.diag(np.ones(size - 1), k=1) + np.diag(np.ones(size - 1), k=-1)
    exchange -= np.diag(np.ones(size - n), k=n) + np.diag(np.ones(size - n), k=-n)
    return sp.csr_matrix(exchange.astype(np.int32))


def create_cylinder_exchange(threshold):
    exchange = threshold*np.identity(size)
    exchange -= np.diag(np.ones(size - 1), k=1) + np.diag(np.ones(size - 1), k=-1)
    exchange -= np.diag(np.ones(size - n), k=n) + np.diag(np.ones(size - n), k=-n)
    exchange -= np.diag(np.ones(n), k=size-n) + np.diag(np.ones(n), k=-size+n)
    return sp.csr_matrix(exchange.astype(np.int32))


def topple(sandpile, exchange):
    while True:
        unstable = sandpile >= 4
        if not unstable.any():
            break
        sandpile -= exchange @ unstable.astype(np.int32)
    return sandpile.astype(np.int32)


def add_some(sandpile):
    new_sand = np.zeros(size).astype(np.int32)
    new_sand[np.random.randint(0, size)] = 4
    sandpile += new_sand
    return sandpile.astype(np.int32)


def add_lots(sandpile):
    prob = 0.3
    new_sand = np.random.rand(size) < prob
    sandpile += new_sand
    return sandpile


start_time = time.time()

# Create the grid
sandpile = np.zeros(size)

# Generate the exchange matrix
exchange = create_cylinder_exchange(threshold)

for i in range(nsteps):
    # Update the sandpile to add new sand
    sandpile = add_some(sandpile)
    # print(sandpile.reshape((n, n)))
    # Initiate avalanche
    sandpile = topple(sandpile, exchange)
    # print(sandpile.reshape((n, n)))

end_time = time.time()

elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time, " seconds")
