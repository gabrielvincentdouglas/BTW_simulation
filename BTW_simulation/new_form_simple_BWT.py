import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.sparse as sp
import random
from matplotlib.animation import FuncAnimation

# Parameters
n = 100  # Grid size is n*n
size = n**2  # Total number of cells
centre = size//2
threshold = 4  # Threshold value for sandpile
nsteps = 1000  # Number of simulation steps
gradient = 10  # Specify how steep the slope is
even_weight = np.ones(4)  # Unbiased toppling to adjacent cells
slope_weight = gradient*np.array([2, 3, 2, 1])
# Can change tilt-axis of slope, higher value means greater tilt down in this direction [south, east, north, west]
standard = 0
cylindrical = 1


# Function to create the simple exchange matrix for a 2d grid such that sand distributes evenly to directly adjacent
# cells. Specify type standard for standard grid or cylindrical for cylindrical grid
def create_exchange(threshold, type):
    # Add the amount to be taken out from toppled cell
    exchange = threshold * np.identity(size)
    # Add connections for cells adjacent in the grid
    exchange -= np.diag(np.ones(size - 1), k=1) + np.diag(np.ones(size - 1), k=-1)
    exchange -= np.diag(np.ones(size - n), k=n) + np.diag(np.ones(size - n), k=-n)
    if type == 0:
        for i in range(1, size - 1):
            if i % n == 0:
                exchange[i - 1, i] += 1
            elif i % n == n - 1:
                exchange[i + 1, i] += 1
    return sp.csr_matrix(exchange.astype(np.int32))


# Function distributes four units across an array of length 4 with probability given by the input weight vector
def random_sand(sand_fall, weight):
    # Randomly select with repetition which adjacent cells collect toppled sand
    a, b, c, d = random.choices(range(4), weight, k=4)
    sand_fall[a] += 1
    sand_fall[b] += 1
    sand_fall[c] += 1
    sand_fall[d] += 1
    return sand_fall


# Function to topple unstable cells and distribute sand to adjacent cells until sandpile stabilises
def topple(sandpile, exchange, topple_record):
    while (sandpile >= threshold).any():
        unstable = sandpile >= 4
        topple_record += unstable
        sandpile -= exchange @ unstable.astype(np.int32)
    topple_counts.append(np.sum(topple_record))
    area_counts.append(np.sum(topple_record >= 1) / size)
    return sandpile.astype(np.int32), topple_record


def simple_topple(sandpile, topple_record):
    while (sandpile >= threshold).any():
        unstable = sandpile >= 4
        for i in range(size):
            if sandpile[i] >= threshold:
                sandpile[i] -= 4
                if i == 0:
                    sandpile[i + 1] += 1
                    sandpile[i + n] += 1
                elif i == n:
                    sandpile[i - 1] += 1
                    sandpile[i + n] += 1
                elif i == size - 1 - n:
                    sandpile[i + 1] += 1
                    sandpile[i - n] += 1
                elif i == size - 1:
                    sandpile[i - 1] += 1
                    sandpile[i - n] += 1
                elif i in range(1, n):
                    sandpile[i + 1] += 1
                    sandpile[i + n] += 1
                    sandpile[i - 1] += 1
                elif i % n == 0:
                    sandpile[i + 1] += 1
                    sandpile[i + n] += 1
                    sandpile[i - n] += 1
                elif i % n == n - 1:
                    sandpile[i - 1] += 1
                    sandpile[i + n] += 1
                    sandpile[i - n] += 1
                elif i in range(size - 1 - n, size - 1):
                    sandpile[i + 1] += 1
                    sandpile[i - n] += 1
                    sandpile[i - 1] += 1
                else:
                    sandpile[i + 1] += 1
                    sandpile[i + n] += 1
                    sandpile[i - 1] += 1
                    sandpile[i - n] += 1

    return sandpile.astype(np.int32), topple_record


# Function to topple unstable cells and distribute the sand to adjacent cells randomly using random_sand and loops
# until grid stabilises
def random_topple(sandpile, topple_record, weight):
    while (sandpile >= threshold).any():
        unstable = sandpile >= 4
        for i in range(size):
            if sandpile[i] >= threshold:
                sandpile[i] -= 4
                sand_fall = [0, 0, 0, 0]
                random_sand(sand_fall, weight)
                if i == 0:
                    sandpile[i + 1] += sand_fall[1]
                    sandpile[i + n] += sand_fall[0]
                elif i == n:
                    sandpile[i - 1] += sand_fall[3]
                    sandpile[i + n] += sand_fall[0]
                elif i == size - 1 - n:
                    sandpile[i + 1] += sand_fall[1]
                    sandpile[i - n] += sand_fall[2]
                elif i == size - 1:
                    sandpile[i - 1] += sand_fall[3]
                    sandpile[i - n] += sand_fall[2]
                elif i in range(1, n):
                    sandpile[i + 1] += sand_fall[1]
                    sandpile[i + n] += sand_fall[0]
                    sandpile[i - 1] += sand_fall[3]
                elif i % n == 0:
                    sandpile[i + 1] += sand_fall[1]
                    sandpile[i + n] += sand_fall[0]
                    sandpile[i - n] += sand_fall[2]
                elif i % n == n - 1:
                    sandpile[i - 1] += sand_fall[3]
                    sandpile[i + n] += sand_fall[0]
                    sandpile[i - n] += sand_fall[2]
                elif i in range(size - 1 - n, size - 1):
                    sandpile[i + 1] += sand_fall[1]
                    sandpile[i - n] += sand_fall[2]
                    sandpile[i - 1] += sand_fall[3]
                else:
                    sandpile[i + 1] += sand_fall[1]
                    sandpile[i + n] += sand_fall[0]
                    sandpile[i - 1] += sand_fall[3]
                    sandpile[i - n] += sand_fall[2]

    return sandpile.astype(np.int32), topple_record


# Function to topple unstable cells and distribute the sand randomly with weight determined by the height adjacent
# cells - in addition to input weighting - using random_sand and loops process until the grid stabilises
def relative_random_topple(sandpile, topple_record, weighting):
    while True:
        unstable = sandpile >= threshold
        topple_record += unstable
        if not unstable.any():
            break
        for i in range(size):
            if sandpile[i] >= threshold:
                if i == 0:
                    diff = sandpile[i] - np.array([sandpile[i + n], sandpile[i + 1], 0, 0])
                    weight = np.maximum(diff, 0) + weighting
                    sand_fall = [0, 0, 0, 0]
                    random_sand(sand_fall, weight)
                    sandpile[i + n] += sand_fall[0]
                    sandpile[i + 1] += sand_fall[1]
                elif i == n:
                    diff = sandpile[i] - np.array([sandpile[i + n], 0, 0, sandpile[i - 1]])
                    weight = np.maximum(diff, 0) + weighting
                    sand_fall = [0, 0, 0, 0]
                    random_sand(sand_fall, weight)
                    sandpile[i + n] += sand_fall[0]
                    sandpile[i - 1] += sand_fall[3]
                elif i == size - 1 - n:
                    diff = sandpile[i] - np.array([0, sandpile[i + 1], sandpile[i - n], 0])
                    weight = np.maximum(diff, 0) + weighting
                    sand_fall = [0, 0, 0, 0]
                    random_sand(sand_fall, weight)
                    sandpile[i + 1] += sand_fall[1]
                    sandpile[i - n] += sand_fall[2]
                elif i == size - 1:
                    diff = sandpile[i] - np.array([0, 0, sandpile[i - n], sandpile[i - 1]])
                    weight = np.maximum(diff, 0) + weighting
                    sand_fall = [0, 0, 0, 0]
                    random_sand(sand_fall, weight)
                    sandpile[i - n] += sand_fall[2]
                    sandpile[i - 1] += sand_fall[3]
                elif i in range(1, n):
                    diff = sandpile[i] - np.array([sandpile[i + n], sandpile[i + 1], 0, sandpile[i - 1]])
                    weight = np.maximum(diff, 0) + weighting
                    sand_fall = [0, 0, 0, 0]
                    random_sand(sand_fall, weight)
                    sandpile[i + n] += sand_fall[0]
                    sandpile[i + 1] += sand_fall[1]
                    sandpile[i - 1] += sand_fall[3]
                elif i % n == 0:
                    diff = sandpile[i] - np.array(sandpile[i + n], [sandpile[i + 1], sandpile[i - n], 0])
                    weight = np.maximum(diff, 0) + weighting
                    sand_fall = [0, 0, 0, 0]
                    random_sand(sand_fall, weight)
                    sandpile[i + n] += sand_fall[0]
                    sandpile[i + 1] += sand_fall[1]
                    sandpile[i - n] += sand_fall[2]
                elif i % n == n - 1:
                    diff = sandpile[i] - np.array([sandpile[i + n], 0, sandpile[i - n], sandpile[i - 1], ])
                    weight = np.maximum(diff, 0) + weighting
                    sand_fall = [0, 0, 0, 0]
                    random_sand(sand_fall, weight)
                    sandpile[i + n] += sand_fall[0]
                    sandpile[i - n] += sand_fall[2]
                    sandpile[i - 1] += sand_fall[3]
                elif i in range(size - 1 - n, size - 1):
                    diff = sandpile[i] - np.array([0, sandpile[i + 1], sandpile[i - n], sandpile[i - 1]])
                    weight = np.maximum(diff, 0) + weighting
                    sand_fall = [0, 0, 0, 0]
                    random_sand(sand_fall, weight)
                    sandpile[i + 1] += sand_fall[1]
                    sandpile[i - n] += sand_fall[2]
                    sandpile[i - 1] += sand_fall[3]
                else:
                    diff = sandpile[i] - np.array([sandpile[i + n], sandpile[i + 1], sandpile[i - n], sandpile[i - 1]])
                    weight = np.maximum(diff, 0) + weighting
                    sand_fall = [0, 0, 0, 0]
                    random_sand(sand_fall, weight)
                    sandpile[i + n] += sand_fall[0]
                    sandpile[i + 1] += sand_fall[1]
                    sandpile[i - n] += sand_fall[2]
                    sandpile[i - 1] += sand_fall[3]
                sandpile[i] -= 4
    return sandpile.astype(np.int32), topple_record


# Functions to add an amount of sand
def add_some(sandpile, amount):
    new_sand = np.zeros(size).astype(np.int32)
    new_sand[np.random.randint(0, size)] = amount
    sandpile += new_sand
    return sandpile.astype(np.int32), new_sand


def add_centre(sandpile, amount):
    new_sand = np.zeros(size).astype(np.int32)
    new_sand[centre] = amount
    sandpile += new_sand
    return sandpile.astype(np.int32), new_sand


def add_lots(sandpile, prob):
    new_sand = np.random.rand(size) < prob
    new_sand = new_sand.astype(np.int32)
    sandpile += new_sand
    return sandpile.astype(np.int32), new_sand


# Create the grid
sandpile = np.zeros(size)
sand_fall = np.zeros(4)

# Initiate storage of statistics
sand_tally = np.zeros(size)
topple_record = np.zeros(size)
topple_counts = []
area_counts = []

# Generate the exchange matrix
exchange = create_exchange(threshold)


def simulate(sandpile):
    global sand_tally, topple_record
    # Run the simulation for the specified number of steps
    for i in range(nsteps):
        # Add some sand to the grid
        sandpile, new_sand = add_some(sandpile, 4)
        sand_tally += new_sand
        # print(sandpile.reshape((n, n)))
        # Initiate toppling
        sandpile, topple_record = topple(sandpile, exchange, topple_record)
        # print(sandpile.reshape((n, n)))
    # Track which unique cells toppled over total time
    topple_map = topple_record >= 1
    # Calculate the area of an avalanche proportional to grid area
    area = np.sum(topple_map) / (size * size)
    # Calculate sand loss
    loss = np.sum(sand_tally) - np.sum(sandpile)

    # Create a figure with two subplots, the sandpile distribution and the avalanche map
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the first image on the left subplot
    im1 = axs[0].imshow(sandpile.reshape((n, n)), cmap='Greys', vmin=0, vmax=threshold, interpolation='nearest')
    axs[0].set_title(f"Step {nsteps}", color='black')
    fig.colorbar(im1, ax=axs[0])

    # Plot the second image on the right subplot
    axs[1].imshow(topple_map.reshape((n, n)), cmap='Greys', vmin=0, vmax=3, interpolation='nearest')
    axs[1].set_title("Map of cells affected by avalanche", color='black')

    # Show the plot
    plt.show()


# Define a function to update the plot for each frame
def animate(frame):
    global sandpile, topple_record
    # Add to the grid 'x' amount of sand
    sandpile, new_sand = add_lots(sandpile, 0.01)

    # Initiate toppling
    sandpile, topple_record = topple(sandpile, exchange, np.zeros(size))

    # Update the plot
    im1.set_data(sandpile.reshape((n, n)))
    im2.set_data(topple_record.reshape((n, n)))

    # Return the plot objects to be updated
    return im1, im2


# Create the figure and plot the initial state
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
im1 = axs[0].imshow(sandpile.reshape((n, n)), cmap='Greys', vmin=0, vmax=threshold, interpolation='nearest')
axs[0].set_title("Sandpile")
fig.colorbar(im1, ax=axs[0])
im2 = axs[1].imshow(topple_record.reshape((n, n)), cmap='Greys', vmin=0, vmax=3, interpolation='nearest')
axs[1].set_title("Map of cells affected by avalanche", color='black')

# Create the animation object
ani = FuncAnimation(fig, animate, frames=nsteps, interval=10, blit=True, repeat=False)

# Display the animation
plt.show()

