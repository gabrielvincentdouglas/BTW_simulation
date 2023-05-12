import numpy as np
import time
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt


# Parameters
threshold = 4
n = 100
size = n**2
centre = size//2 + n//2
steps = 1


def create_exchange(threshold):
    exchange = threshold*np.identity(size)
    exchange -= np.diag(np.ones(size - 1), k=1) + np.diag(np.ones(size - 1), k=-1)
    exchange -= np.diag(np.ones(size - n), k=n) + np.diag(np.ones(size - n), k=-n)
    for i in range(1, size - 1):
        if i % n == 0:
            exchange[i, i - 1] = 0
        elif i % n == n - 1:
            exchange[i, i + 1] = 0
    return csr_matrix(exchange.astype(int))


def topple(sandpile, exchange):
    while (sandpile >= threshold).any():
        unstable = sandpile >= threshold
        sandpile -= unstable.astype(int) @ exchange
    return sandpile


def add_some(sandpile, amount):
    new_sand = np.zeros(size)
    new_sand[np.random.randint(0, size)] += amount
    sandpile += new_sand.astype(int)
    return sandpile.astype(int)


start_time = time.time()

# Initialise sandpile
sandpile = np.zeros(size)
exchange = create_exchange(threshold)


# Simulate sandpile evolution
for i in range(steps):
    # sandpile = add_some(sandpile, 4)
    sandpile[centre] += 30000
    # print(sandpile.reshape((n, n)))
    sandpile = topple(sandpile, exchange)
    # print(sandpile.reshape((n, n)))

end_time = time.time()  # Record the end time

elapsed_time = end_time - start_time  # Calculate the elapsed time

print("Elapsed time: {:.2f} seconds".format(elapsed_time))

# Configure the plot
fig, ax = plt.subplots()
im = ax.imshow(sandpile.reshape((n, n)), cmap='Greys', vmin=0, vmax=threshold, interpolation='nearest', origin='lower')
fig.colorbar(im, ax=ax)
ax.set_title("Sandpile")

# plt.show()
