import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from matplotlib.animation import FuncAnimation

# Parameters
size = 51
threshold = 4
prob = 0.01
nsteps = 3000


def update(grid):
    # Add sand
    new_sand = np.zeros((size, size))
    new_sand[25, 25] = 3
    # Update grid
    grid += new_sand
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
    grid = update(grid)
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
anim = FuncAnimation(fig, animate, frames=1000, interval=1, blit=True, repeat=False)

# Show the animation
plt.show()
