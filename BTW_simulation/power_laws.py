import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import powerlaw
import time

# Parameters
threshold = 4
n = 50
steps = 100000


def topple(sandpile):
    topple_record = np.zeros((n, n))
    avl_time = 0
    while (sandpile >= threshold).any():
        unstable = sandpile >= threshold
        avl_time += 1
        topple_record += unstable
        exchange = convolve2d(unstable, kernel, mode='same')
        sandpile += exchange
    return sandpile, topple_record, avl_time


def add_some(sandpile, amount):
    new_sand = np.zeros((n, n))
    rand_index = np.random.randint(n), np.random.randint(n)
    new_sand[rand_index] += amount
    sandpile += new_sand
    return sandpile


# Simulate sandpile evolution
def add_and_topple(sandpile, steps, amount):
    topples = []
    areas = []
    avl_times = []
    for i in range(steps):
        sandpile = add_some(sandpile, amount)

        sandpile, topple_record, avl_time = topple(sandpile)

        topple_map = topple_record > 0

        topples.append(np.sum(topple_record))
        areas.append(np.sum(topple_map))
        avl_times.append(avl_time)
    return sandpile, np.array(topples), np.array(areas), np.array(avl_times)


start_time = time.time()

# Initialise sandpile
kernel = np.array([[0, 1, 0],
                   [1, -threshold, 1],
                   [0, 1, 0]])
sandpile = 30 * np.ones((n, n))
sandpile = topple(sandpile)[0]

sandpile, topples, area, avl_times = add_and_topple(sandpile, 10000, 1)

end_time = time.time()

print('Duration is', end_time - start_time)

# Fit the data to a power law distribution
data = [topples[topples > 0], area[area > 0], avl_times[avl_times > 0]]
fit = [0, 0, 0]
for i in range(3):
    fit[i] = powerlaw.Fit(data[i], xmin=min(data[i]), discrete=True)


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


