import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from matplotlib.animation import FuncAnimation

# Parameters
size = 50
threshold = 4
nsteps = 1000


def add_one(grid):
    # Generate an array to add one grain of sand at random
    new_sand = np.zeros((size, size))
    # Choose a random row and column index
    row = np.random.randint(0, size)
    col = np.random.randint(0, size)
    # Set the randomly chosen entry to 1
    new_sand[row, col] = 2
    grid += new_sand
    return grid


def update(grid, prob):
    # Add sand to random cells with probability prob
    new_sand = np.random.rand(size, size) < prob
    grid += new_sand.astype(int)

    return grid


def topple(grid):
    # Identify the unstable cells
    unstable = grid >= threshold
    # Create the convolution kernel to identify adjacent (vertical and horizontal) unstable cells
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    # Identify sand to move, 1 grain from each adjacent unstable cell
    sand_there = scipy.signal.convolve2d(unstable, kernel, mode='same')
    # Enact toppling
    grid += sand_there - threshold*unstable
    return grid, unstable.astype(int)


# Initialise grid
grid = np.zeros((size, size))
prev_grid = None

# Initialise storage of statistics
topple_record = np.zeros((size, size))
topple_counts = []  # initialise list to record number of topples per iteration
topple_map = np.zeros((size, size))

# Set up the figure and axes
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left subplot
im1 = axs[0].imshow(grid, cmap='Greys', vmin=0, vmax=threshold, interpolation='nearest')
axs[0].set_title("Sandpile model")

# Plot the second image on the right subplot
im2 = axs[1].imshow(topple_map, cmap='Greys', vmin=0, vmax=3, interpolation='nearest')
axs[1].set_title("Map of cells affected by avalanche", color='black')


# Define the animation function
def animate(i):
    global grid, prev_grid, topple_record
    grid = add_one(grid)
    # Simulate avalanche
    while not np.array_equal(grid, prev_grid):
        prev_grid = grid.copy()
        grid, unstable = topple(grid)
        topple_record += unstable
        topple_map = topple_record >= 1
    im1.set_array(grid)
    im2.set_array(topple_map)
    return [im1, im2]


# Create the animation
anim = FuncAnimation(fig, animate, frames=500, interval=5, blit=True, repeat=False)

# Show the animation
plt.show()
