import numpy as np
import random
import pandas as pd
import powerlaw
import matplotlib.pyplot as plt

# Define parameters for the sandpile simulation
threshold = 4  # Sand grains topple when the height exceeds this threshold
n = 50  # Grid size (n x n)
size = n ** 2  # Total number of cells in the grid
steps = 10000  # Number of iterations to run the simulation
gradient = 1  # The size of the slope of the sandpile
tilt = gradient * np.array([1, 1, 1, 1])  # Directions to topple sand [north, east, south, west], increasing value
# increases likelihood of sand falling in corresponding direction
amount = 1  # The amount of sand to add to the pile at each iteration


# Define a function to randomly select neighboring cells to receive toppled sand
def random_sand(weight):
    sand_fall = [0, 0, 0, 0]
    indices = random.choices(range(4), weight, k=4)  # Randomly choose with repetition from 4 directions
    for i in indices:
        sand_fall[i] += 1  # Assign the sand to the corresponding direction
    return sand_fall


# Define a function to simulate toppling of sand at each cell of the grid
# Toppling to neighbouring cells is weighted by the height difference and the tilt parameter
def relative_topple(sandpile, tilt):
    topple_record = np.zeros(size)   # Keep track of how many times each cell has toppled
    avl_time = 0    # Keep track of the number of iterations taken to stabilize the sandpile
    while (sandpile >= threshold).any():
        avl_time += 1   # Increment the number of iterations taken
        topple_record += sandpile >= threshold   # Add to the topple record for cells that topple
        for i in range(size):
            if sandpile[i] >= threshold:
                if i == 0:
                    diff = sandpile[i] - np.array([sandpile[i + n], sandpile[i + 1], 0, 0])
                    weight = np.maximum(diff, 0) + tilt
                    sand_fall = random_sand(weight)
                    sandpile[i + n] += sand_fall[0]
                    sandpile[i + 1] += sand_fall[1]
                elif i == n - 1:
                    diff = sandpile[i] - np.array([sandpile[i + n], 0, 0, sandpile[i - 1]])
                    weight = np.maximum(diff, 0) + tilt
                    sand_fall = random_sand(weight)
                    sandpile[i + n] += sand_fall[0]
                    sandpile[i - 1] += sand_fall[3]
                elif i == size - n:
                    diff = sandpile[i] - np.array([0, sandpile[i + 1], sandpile[i - n], 0])
                    weight = np.maximum(diff, 0) + tilt
                    sand_fall = random_sand(weight)
                    sandpile[i + 1] += sand_fall[1]
                    sandpile[i - n] += sand_fall[2]
                elif i == size - 1:
                    diff = sandpile[i] - np.array([0, 0, sandpile[i - n], sandpile[i - 1]])
                    weight = np.maximum(diff, 0) + tilt
                    sand_fall = random_sand(weight)
                    sandpile[i - n] += sand_fall[2]
                    sandpile[i - 1] += sand_fall[3]
                elif i in range(n):
                    diff = sandpile[i] - np.array([sandpile[i + n], sandpile[i + 1], 0, sandpile[i - 1]])
                    weight = np.maximum(diff, 0) + tilt
                    sand_fall = random_sand(weight)
                    sandpile[i + n] += sand_fall[0]
                    sandpile[i + 1] += sand_fall[1]
                    sandpile[i - 1] += sand_fall[3]
                elif i % n == 0:
                    diff = sandpile[i] - np.array([sandpile[i + n], sandpile[i + 1], sandpile[i - n], 0])
                    weight = np.maximum(diff, 0) + tilt
                    sand_fall = random_sand(weight)
                    sandpile[i + n] += sand_fall[0]
                    sandpile[i + 1] += sand_fall[1]
                    sandpile[i - n] += sand_fall[2]
                elif i % n == n - 1:
                    diff = sandpile[i] - np.array([sandpile[i + n], 0, sandpile[i - n], sandpile[i - 1]])
                    weight = np.maximum(diff, 0) + tilt
                    sand_fall = random_sand(weight)
                    sandpile[i + n] += sand_fall[0]
                    sandpile[i - n] += sand_fall[2]
                    sandpile[i - 1] += sand_fall[3]
                elif i in range(size - n, size):
                    diff = sandpile[i] - np.array([0, sandpile[i + 1], sandpile[i - n], sandpile[i - 1]])
                    weight = np.maximum(diff, 0) + tilt
                    sand_fall = random_sand(weight)
                    sandpile[i + 1] += sand_fall[1]
                    sandpile[i - n] += sand_fall[2]
                    sandpile[i - 1] += sand_fall[3]
                else:
                    diff = sandpile[i] - np.array([sandpile[i + n], sandpile[i + 1], sandpile[i - n], sandpile[i - 1]])
                    weight = np.maximum(diff, 0) + tilt
                    sand_fall = random_sand(weight)
                    sandpile[i + n] += sand_fall[0]
                    sandpile[i + 1] += sand_fall[1]
                    sandpile[i - n] += sand_fall[2]
                    sandpile[i - 1] += sand_fall[3]
                sandpile[i] -= threshold

    return sandpile, topple_record, avl_time


def add_some(sandpile, amount):
    new_sand = np.zeros(size)
    new_sand[np.random.randint(0, size)] += amount
    sandpile += new_sand.astype(int)
    return sandpile.astype(int)


# Simulate sandpile evolution
def add_and_topple(sandpile, steps, amount, tilt):
    topples = []
    areas = []
    avl_times = []
    for i in range(steps):
        sandpile = add_some(sandpile, amount)

        sandpile, topple_record, avl_time = relative_topple(sandpile, tilt)

        topple_map = topple_record > 0

        topples.append(np.sum(topple_record))
        areas.append(np.sum(topple_map))
        avl_times.append(avl_time)
    return sandpile, pd.Series(topples), pd.Series(areas), pd.Series(avl_times)


sandpile = 20 * np.ones(size)
sandpile = relative_topple(sandpile, tilt)[0]

sandpile, topples, area, avl_times = add_and_topple(sandpile, steps, 1, tilt)

# Fit the data to a power law distribution
data = [topples[topples > 0], area[area > 0], avl_times[avl_times > 0]]
fit = [0, 0, 0]
for i in range(3):
    x = data[i].index.values
    y = data[i].values
    fit[i] = powerlaw.Fit(y, xmin=min(y), discrete=True)

# Configure the plot
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
titles = ["Number of topples", "Area of avalanche", "Avalanche duration"]

for i in range(3):
    fit[i].power_law.plot_pdf(
        color='gray',
        linestyle='--',
        linewidth=0.8,
        ax=ax[i],
        label=r'$t^{{-\alpha}}: \alpha$={:.2f}'.format(fit[i].alpha))
    fit[i].plot_pdf(
        color='k',
        ax=ax[i],
        linewidth=0.8,
        label="PDF")

    ax[i].set_xlabel(titles[i])
    ax[i].legend()

plt.show()
