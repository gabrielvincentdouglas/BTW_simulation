import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

# Parameters
size = 50
threshold = 4
prob = 0.01
nsteps = 100


def update(grid):
    # Add sand to random cells with probability prob
    new_sand = np.random.rand(size, size) < prob
    grid += new_sand
    return grid


def relative_topple(grid):
    # Create a copy of the input to store the updated values
    new_grid = grid.copy()
    # Loop over each cell in the array
    for i in range(size):
        for j in range(size):
            # Check if the current cell has a value of 4
            if grid[i, j] == threshold:
                # Check if the current cell is on the border
                if i == 0 or i == size - 1 or j == 0 or j == size - 1:
                    # If so, subtract 1 from the current cell's value
                    new_grid[i, j] -= 1
                else:
                    # Otherwise, add 1 to each adjacent cell that is 4 less than the current cell
                    # and subtract 1 from the current cell
                    if i > 0 and grid[i - 1, j] == grid[i, j] - 4:
                        new_grid[i - 1, j] += 1
                        new_grid[i, j] -= 1
                    if i < size - 1 and grid[i + 1, j] == grid[i, j] - 4:
                        new_grid[i + 1, j] += 1
                        new_grid[i, j] -= 1
                    if j > 0 and grid[i, j - 1] == grid[i, j] - 4:
                        new_grid[i, j - 1] += 1
                        new_grid[i, j] -= 1
                    if j < size - 1 and grid[i, j + 1] == grid[i, j] - 4:
                        new_grid[i, j + 1] += 1
                        new_grid[i, j] -= 1

    return new_grid


# Initialise grid
grid = np.zeros((size, size))
prev_grid = None

# Simulate the model
for i in range(nsteps):
    grid = update(grid)
    # Simulate avalanche
    while not np.array_equal(grid, prev_grid):
        prev_grid = grid.copy()
        grid = relative_topple(grid)


plt.imshow(grid, cmap='Greys', vmin=0, vmax=threshold, interpolation='nearest')
plt.title(f"Step {nsteps - 1}", color='black')
plt.colorbar()
plt.show()