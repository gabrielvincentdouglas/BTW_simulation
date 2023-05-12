import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

n = 100
threshold = 4


def topple(sandpile):
    while (sandpile >= threshold).any():
        unstable = sandpile >= threshold
        exchange = convolve2d(unstable, kernel, mode='same')
        sandpile += exchange
    return sandpile


kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

# Create a minimally stable sandpile
sandpile = 0 * np.ones((n, n))
sandpile[50, 50] = 100000
sandpile = topple(sandpile)

# Create a dictionary recording the cells at each height level
levels = {0: None, 1: None, 2: None, 3: None}
for i in range(4):
    mask = (sandpile == i)
    levels[i] = mask.astype(int)


def box_count(level):
    # Find all the non-zero cells
    cells = np.argwhere(level > 0)
    # Compute the fractal dimension using logarithmic scales
    scales = np.logspace(1, 6, num=10, endpoint=False, base=2)
    counts = []
    # Box count over scales using histograms
    for scale in scales:
        # Compute the histogram
        H, edges = np.histogramdd(cells, bins=(np.arange(0, n, scale), np.arange(0, n, scale)))
        counts.append(np.sum(H > 0))
    # linear fit, polynomial of degree 1
    coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
    return scales, counts, coeffs


dims = []

fig, axs = plt.subplots(1, 4)

for i, ax in enumerate(axs.flatten()):
    if i < 4:
        ax.imshow(levels[i], cmap='binary', interpolation='nearest', origin='lower')
        ax.set_title(f"Height level {i}")
        ax.axis('off')
        coeffs = box_count(levels[i])[2]
        dim = -coeffs[0]
        dims.append(dim)

        print(dim)


np.savetxt('H_dimension.csv', dims, delimiter=',', newline=';')
plt.savefig('fractals_2.pdf')

