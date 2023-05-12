import numpy as np
import random
import time

# Parameters
threshold = 4
n = 3
size = n**2


def random_sand(weight):
    sand_fall = [0, 0, 0, 0]
    # Randomly select with repetition which adjacent cells collect toppled sand
    indices = random.choices(range(4), weight, k=4)
    for i in indices:
        sand_fall[i] += 1
    return sand_fall


def relative_topple(sandpile, weighting):
    while (sandpile >= threshold).any():
        for i in range(size):
            if sandpile[i] >= threshold:
                if i == 0:
                    diff = sandpile[i] - np.array([sandpile[i + n], sandpile[i + 1], 0, 0])
                    weight = np.maximum(diff, 0) + weighting
                    sand_fall = random_sand(weight)
                    sandpile[i + n] += sand_fall[0]
                    sandpile[i + 1] += sand_fall[1]
                elif i == n - 1:
                    diff = sandpile[i] - np.array([sandpile[i + n], 0, 0, sandpile[i - 1]])
                    weight = np.maximum(diff, 0) + weighting
                    sand_fall = random_sand(weight)
                    sandpile[i + n] += sand_fall[0]
                    sandpile[i - 1] += sand_fall[3]
                elif i == size - n:
                    diff = sandpile[i] - np.array([0, sandpile[i + 1], sandpile[i - n], 0])
                    weight = np.maximum(diff, 0) + weighting
                    sand_fall = random_sand(weight)
                    sandpile[i + 1] += sand_fall[1]
                    sandpile[i - n] += sand_fall[2]
                elif i == size - 1:
                    diff = sandpile[i] - np.array([0, 0, sandpile[i - n], sandpile[i - 1]])
                    weight = np.maximum(diff, 0) + weighting
                    sand_fall = random_sand(weight)
                    sandpile[i - n] += sand_fall[2]
                    sandpile[i - 1] += sand_fall[3]
                elif i in range(n):
                    diff = sandpile[i] - np.array([sandpile[i + n], sandpile[i + 1], 0, sandpile[i - 1]])
                    weight = np.maximum(diff, 0) + weighting
                    sand_fall = random_sand(weight)
                    sandpile[i + n] += sand_fall[0]
                    sandpile[i + 1] += sand_fall[1]
                    sandpile[i - 1] += sand_fall[3]
                elif i % n == 0:
                    diff = sandpile[i] - np.array([sandpile[i + n], sandpile[i + 1], sandpile[i - n], 0])
                    weight = np.maximum(diff, 0) + weighting
                    sand_fall = random_sand(weight)
                    sandpile[i + n] += sand_fall[0]
                    sandpile[i + 1] += sand_fall[1]
                    sandpile[i - n] += sand_fall[2]
                elif i % n == n - 1:
                    diff = sandpile[i] - np.array([sandpile[i + n], 0, sandpile[i - n], sandpile[i - 1]])
                    weight = np.maximum(diff, 0) + weighting
                    sand_fall = random_sand(weight)
                    sandpile[i + n] += sand_fall[0]
                    sandpile[i - n] += sand_fall[2]
                    sandpile[i - 1] += sand_fall[3]
                elif i in range(size - n, size):
                    diff = sandpile[i] - np.array([0, sandpile[i + 1], sandpile[i - n], sandpile[i - 1]])
                    weight = np.maximum(diff, 0) + weighting
                    sand_fall = random_sand(weight)
                    sandpile[i + 1] += sand_fall[1]
                    sandpile[i - n] += sand_fall[2]
                    sandpile[i - 1] += sand_fall[3]
                else:
                    diff = sandpile[i] - np.array([sandpile[i + n], sandpile[i + 1], sandpile[i - n], sandpile[i - 1]])
                    weight = np.maximum(diff, 0) + weighting
                    sand_fall = random_sand(weight)
                    sandpile[i + n] += sand_fall[0]
                    sandpile[i + 1] += sand_fall[1]
                    sandpile[i - n] += sand_fall[2]
                    sandpile[i - 1] += sand_fall[3]
                sandpile[i] -= threshold
    return sandpile


def add_some(sandpile, amount):
    new_sand = np.zeros(size)
    new_sand[np.random.randint(0, size)] += amount
    sandpile += new_sand.astype(int)
    return sandpile.astype(int)


# Simulate sandpile evolution
def add_and_topple(sandpile, steps, amount, tilt):
    for i in range(steps):
        sandpile = add_some(sandpile, amount)
        # print(sandpile.reshape((n, n)))
        sandpile = relative_topple(sandpile, tilt)
        # print(sandpile.reshape((n, n)))
    return sandpile


start_time = time.time()


steps = 1000
gradient = 1
tilt = gradient*np.array([1, 1, 1, 1])  # [north, east, south, west]
amount = 1


# Initialise sandpile
sandpile = 2*np.ones(size)
topple_record = np.zeros(size)

add_and_topple(sandpile, steps, 1, tilt)

end_time = time.time()  # Record the end time

elapsed_time = end_time - start_time  # Calculate the elapsed time

print("Elapsed time: {:.2f} seconds".format(elapsed_time))
