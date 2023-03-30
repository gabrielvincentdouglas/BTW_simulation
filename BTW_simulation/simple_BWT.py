import numpy as np
import matplotlib.pyplot as plt
import time
import random

# Parameters
size = 50
threshold = 4
nsteps = 2000


def add_some(grid):
    # Generate an array to add one grain of sand at random
    new_sand = np.zeros((size, size))
    # Choose a random row and column index
    row = np.random.randint(0, size)
    col = np.random.randint(0, size)
    # Set the randomly chosen entry to 1
    new_sand[row, col] = 4
    grid += new_sand
    return grid, new_sand


def add_lots(grid, prob):
    # Add sand to random cells with probability prob
    new_sand = np.random.rand(size, size) < prob
    grid += new_sand.astype(int)

    return grid, new_sand


def topple(grid):
    # Create a copy of the input to store the updated values
    new_grid = grid.copy()
    # Identify unstable cells
    unstable = grid >= 4
    # Loop over each cell in the array
    for i in range(size):
        for j in range(size):
            # Check if the current cell has a value of 4
            if grid[i, j] >= threshold:
                new_grid[i, j] -= 4
                if i == 0 and j == 0:
                    new_grid[i, j + 1] += 1
                    new_grid[i + 1, j] += 1
                elif i == 0 and j == size - 1:
                    new_grid[i + 1, j] += 1
                    new_grid[i, j - 1] += 1
                elif i == size - 1 and j == 0:
                    new_grid[i, j + 1] += 1
                    new_grid[i - 1, j] += 1
                elif i == size - 1 and j == size - 1:
                    new_grid[i, j - 1] += 1
                    new_grid[i - 1, j] += 1
                elif i == 0:
                    new_grid[i, j + 1] += 1
                    new_grid[i + 1, j] += 1
                    new_grid[i, j - 1] += 1
                elif i == size - 1:
                    new_grid[i, j + 1] += 1
                    new_grid[i, j - 1] += 1
                    new_grid[i - 1, j] += 1
                elif j == 0:
                    new_grid[i, j + 1] += 1
                    new_grid[i + 1, j] += 1
                    new_grid[i - 1, j] += 1
                elif j == size - 1:
                    new_grid[i + 1, j] += 1
                    new_grid[i, j - 1] += 1
                    new_grid[i - 1, j] += 1
                else:
                    new_grid[i, j + 1] += 1
                    new_grid[i + 1, j] += 1
                    new_grid[i, j - 1] += 1
                    new_grid[i - 1, j] += 1

    return new_grid, unstable


def cylinder_topple(grid):
    # Create a copy of the input to store the updated values
    new_grid = grid.copy()
    # Identify unstable cells
    unstable = grid >= 4
    # Loop over each cell in the array
    for i in range(size):
        for j in range(size):
            # Check if the current cell has a value of 4
            if grid[i, j] >= threshold:
                new_grid[i, j] -= 4
                if i == 0 and j == 0:
                    new_grid[i, j + 1] += 1
                    new_grid[i + 1, j] += 1
                    new_grid[i, size - 1] += 1
                elif i == 0 and j == size - 1:
                    new_grid[i + 1, j] += 1
                    new_grid[i, j - 1] += 1
                    new_grid[i, 0] += 1
                elif i == size - 1 and j == 0:
                    new_grid[i, j + 1] += 1
                    new_grid[i - 1, j] += 1
                    new_grid[i, size - 1] += 1
                elif i == size - 1 and j == size - 1:
                    new_grid[i, j - 1] += 1
                    new_grid[i - 1, j] += 1
                    new_grid[i, 0] += 1
                elif i == 0:
                    new_grid[i, j + 1] += 1
                    new_grid[i + 1, j] += 1
                    new_grid[i, j - 1] += 1
                elif i == size - 1:
                    new_grid[i, j + 1] += 1
                    new_grid[i, j - 1] += 1
                    new_grid[i - 1, j] += 1
                elif j == 0:
                    new_grid[i, j + 1] += 1
                    new_grid[i + 1, j] += 1
                    new_grid[i - 1, j] += 1
                    new_grid[i, size - 1] += 1
                elif j == size - 1:
                    new_grid[i + 1, j] += 1
                    new_grid[i, j - 1] += 1
                    new_grid[i - 1, j] += 1
                    new_grid[i, 0] += 1
                else:
                    new_grid[i, j + 1] += 1
                    new_grid[i + 1, j] += 1
                    new_grid[i, j - 1] += 1
                    new_grid[i - 1, j] += 1

    return new_grid, unstable


def random_sand(sand_fall):
    # Randomly select with repetition which adjacent cells collect toppled sand
    a, b, c, d = random.choices(range(4), k=4)
    sand_fall[a] += 1
    sand_fall[b] += 1
    sand_fall[c] += 1
    sand_fall[d] += 1
    return sand_fall


def relative_random_sand(sand_fall, weight):
    # Randomly select with repetition which adjacent cells collect toppled sand
    a, b, c, d = random.choices(range(4), weight, k=4)
    sand_fall[a] += 1
    sand_fall[b] += 1
    sand_fall[c] += 1
    sand_fall[d] += 1
    return sand_fall


def random_topple(grid):
    # Create a copy of the input to store the updated values
    new_grid = grid.copy()
    # Identify unstable cells
    unstable = grid >= 4
    # Loop over each cell in the array
    for i in range(size):
        for j in range(size):
            # Check if the current cell has a value of 4
            if grid[i, j] >= threshold:
                new_grid[i, j] -= 4
                sand_fall = [0, 0, 0, 0]
                sand_fall = random_sand(sand_fall)
                if i == 0 and j == 0:
                    new_grid[i, j + 1] += sand_fall[0]
                    new_grid[i + 1, j] += sand_fall[1]
                elif i == 0 and j == size - 1:
                    new_grid[i + 1, j] += sand_fall[1]
                    new_grid[i, j - 1] += sand_fall[2]
                elif i == size - 1 and j == 0:
                    new_grid[i, j + 1] += sand_fall[0]
                    new_grid[i - 1, j] += sand_fall[3]
                elif i == size - 1 and j == size - 1:
                    new_grid[i, j - 1] += sand_fall[2]
                    new_grid[i - 1, j] += sand_fall[3]
                elif i == 0:
                    new_grid[i, j + 1] += sand_fall[0]
                    new_grid[i + 1, j] += sand_fall[1]
                    new_grid[i, j - 1] += sand_fall[2]
                elif i == size - 1:
                    new_grid[i, j + 1] += sand_fall[0]
                    new_grid[i, j - 1] += sand_fall[2]
                    new_grid[i - 1, j] += sand_fall[3]
                elif j == 0:
                    new_grid[i, j + 1] += sand_fall[0]
                    new_grid[i + 1, j] += sand_fall[1]
                    new_grid[i - 1, j] += sand_fall[3]
                elif j == size - 1:
                    new_grid[i + 1, j] += sand_fall[1]
                    new_grid[i, j - 1] += sand_fall[2]
                    new_grid[i - 1, j] += sand_fall[3]
                else:
                    new_grid[i, j + 1] += sand_fall[0]
                    new_grid[i + 1, j] += sand_fall[1]
                    new_grid[i, j - 1] += sand_fall[2]
                    new_grid[i - 1, j] += sand_fall[3]

    return new_grid, unstable


start_time = time.time()

# Initialise grid
grid = np.zeros((size, size))
sand_fall = np.zeros(4)
prev_grid = None

# Initialise storage of statistics
topple_record = np.zeros((size, size))
sand_tally = np.zeros((size, size))
topple_counts = []  # initialise list to record number of topples per iteration
area_counts = []  # initialise list to record the proportional area of an avalanche

# Simulate the model
for i in range(nsteps):
    grid, new_sand = add_some(grid)
    sand_tally += new_sand
    # Simulate avalanche
    while not np.array_equal(grid, prev_grid):
        prev_grid = grid.copy()
        grid, unstable = cylinder_topple(grid)
        topple_record += unstable
    topple_counts.append(np.sum(topple_record))
    area_counts.append(np.sum(topple_record >= 1) / (size*size))

end_time = time.time()

# Track which unique cells toppled over time interval
topple_map = topple_record >= 1
# Calculate the area of an avalanche proportional to grid area
area = np.sum(topple_map) / (size*size)
# Calculate sand loss
loss = np.sum(sand_tally) - np.sum(grid)

elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time, " seconds")
print(f"Average sandpile height at step {nsteps - 1}:", np.mean(grid))
print("The number of cells that toppled at each time step:")
print(topple_counts)
print("The proportional area of avalanche at each time step:")
print(area_counts)
print("The total sand added:", np.sum(sand_tally))
print("The total sand remaining on the grid:", np.sum(grid))
print("The total sand loss:", loss)

# Create a figure with two subplots, the sandpile distribution and the avalanche map
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left subplot
im1 = axs[0].imshow(grid, cmap='Greys', vmin=0, vmax=threshold, interpolation='nearest')
axs[0].set_title(f"Step {nsteps - 1}", color='black')
fig.colorbar(im1, ax=axs[0])

# Plot the second image on the right subplot
axs[1].imshow(topple_map, cmap='Greys', vmin=0, vmax=3, interpolation='nearest')
axs[1].set_title("Map of cells affected by avalanche", color='black')

# Show the plot
plt.show()

