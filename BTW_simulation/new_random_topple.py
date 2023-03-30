import numpy as np
import random
import time

# Parameters
n = 50
size = n**2
threshold = 4
nsteps = 500


def random_sand(sand_fall):
    # Randomly select with repetition which adjacent cells collect toppled sand
    a, b, c, d = random.choices(range(4), k=4)
    sand_fall[a] += 1
    sand_fall[b] += 1
    sand_fall[c] += 1
    sand_fall[d] += 1
    return sand_fall


def create_exchange(threshold):
    exchange = 4*np.identity(size)
    for i in range(size):
        sand_fall = [0, 0, 0, 0]
        random_sand(sand_fall)
        if i == 0:
            exchange[i + 1, i] -= sand_fall[0]
            exchange[i + n, i] -= sand_fall[1]
        elif i in range(1, n):
            exchange[i + 1, i] -= sand_fall[0]
            exchange[i + n, i] -= sand_fall[1]
            exchange[i - 1, i] -= sand_fall[2]
        elif i in range(size - 1 - n, size - 1):
            exchange[i + 1, i] -= sand_fall[0]
            exchange[i - n, i] -= sand_fall[1]
            exchange[i - 1, i] -= sand_fall[2]
        elif i == size - 1:
            exchange[i - 1, i] -= sand_fall[0]
            exchange[i - n, i] -= sand_fall[1]
        else:
            exchange[i + 1, i] -= sand_fall[0]
            exchange[i + n, i] -= sand_fall[1]
            exchange[i - 1, i] -= sand_fall[2]
            exchange[i - n, i] -= sand_fall[3]
    return exchange


def topple(sandpile, exchange):
    for i in range(n**2):
        if sandpile[i] > exchange[i, i]:
            sandpile -= exchange[i, :]
    return sandpile


def avalanche(sandpile):
    prev_sandpile = None
    while not np.array_equal(sandpile, prev_sandpile):
        prev_sandpile = sandpile
        sandpile = topple(sandpile, exchange)
    return sandpile


def add_one(sandpile):
    new_sand = np.zeros(n**2)
    new_sand[np.random.randint(0, n**2)] = 1
    sandpile += new_sand
    return sandpile


def update(sandpile):
    prob = 0.05
    new_sand = np.random.rand(size) < prob
    sandpile += new_sand
    return sandpile


start_time = time.time()

sandpile = np.zeros(n**2)
prev_sandpile = None

for i in range(nsteps):
    # Generate the exchange matrix
    exchange = create_exchange(threshold)
    # Update the sandpile to add new sand
    sandpile = update(sandpile)
    # Initiate avalanche
    while not np.array_equal(sandpile, prev_sandpile):
        prev_sandpile = sandpile
        sandpile = topple(sandpile, exchange)


end_time = time.time()

elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time, " seconds")
print(exchange)