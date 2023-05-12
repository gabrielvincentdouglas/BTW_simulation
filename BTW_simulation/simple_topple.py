import numpy as np
import time

# Parameters
threshold = 4
n = 500
size = n**2
steps = 1


def topple(sandpile):
    while (sandpile >= threshold).any():
        indices = np.argwhere(sandpile >= threshold)
        for i in indices:
            sandpile[i] -= 4
            if i == 0:
                sandpile[i + 1] += 1
                sandpile[i + n] += 1
            elif i == n - 1:
                sandpile[i - 1] += 1
                sandpile[i + n] += 1
            elif i == size - n:
                sandpile[i + 1] += 1
                sandpile[i - n] += 1
            elif i == size - 1:
                sandpile[i - 1] += 1
                sandpile[i - n] += 1
            elif i in range(n):
                sandpile[i + 1] += 1
                sandpile[i + n] += 1
                sandpile[i - 1] += 1
            elif i % n == 0:
                sandpile[i + 1] += 1
                sandpile[i + n] += 1
                sandpile[i - n] += 1
            elif i % n == n - 1:
                sandpile[i + n] += 1
                sandpile[i - 1] += 1
                sandpile[i - n] += 1
            elif i in range(size - n, size):
                sandpile[i + 1] += 1
                sandpile[i - 1] += 1
                sandpile[i - n] += 1
            else:
                sandpile[i + 1] += 1
                sandpile[i + n] += 1
                sandpile[i - 1] += 1
                sandpile[i - n] += 1
    return sandpile


def add_some(sandpile, amount):
    new_sand = np.zeros(size)
    new_sand[np.random.randint(0, size)] += amount
    sandpile += new_sand.astype(int)
    return sandpile.astype(int)


def add_and_topple(sandpile, steps, amount):
    for i in range(steps):
        sandpile = add_some(sandpile, amount)
        # print(sandpile.reshape((n, n)))
        sandpile = topple(sandpile)
        # print(sandpile.reshape((n, n)))


start_time = time.time()

# Initialise sandpile
sandpile = np.zeros(size)

add_and_topple(sandpile, steps, amount=4)

end_time = time.time()  # Record the end time

elapsed_time = end_time - start_time  # Calculate the elapsed time

print("Elapsed time: {:.2f} seconds".format(elapsed_time))

