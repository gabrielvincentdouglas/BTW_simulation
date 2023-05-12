import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import pandas as pd
import powerlaw

# Parameters
threshold = 4
n = 50
steps = 10000


def topple(sandpile):
    topple_record = np.zeros((n, n))
    avl_time = 0
    while True:
        unstable = sandpile >= threshold
        if not unstable.any():
            break
        avl_time += 1
        topple_record += unstable
        exchange = convolve2d(unstable, kernel, mode='same')
        sandpile += exchange
    return sandpile, topple_record, avl_time


def add_some(sandpile, amount):
    new_sand = np.zeros((n, n))
    rand_index = np.random.randint(0, n), np.random.randint(0, n)
    new_sand[rand_index] += amount
    sandpile += new_sand
    return sandpile


# Simulate sandpile evolution
def add_and_topple(sandpile, steps, amount):
    topples = []
    area = []
    avl_times = []
    for i in range(steps):
        sandpile = add_some(sandpile, amount)
        # print(sandpile)
        sandpile, topple_record, avl_time = topple(sandpile)
        # print(sandpile)
        topple_map = topple_record > 0
        topples.append(np.sum(topple_record))
        area.append(np.sum(topple_map))
        avl_times.append(avl_time)
    return sandpile, pd.Series(topples), pd.Series(area), pd.Series(avl_times)


# Initialise sandpile
kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
sandpile = 30 * np.ones((n, n))
sandpile = topple(sandpile)[0]

sandpile, topples, area, avl_times = add_and_topple(sandpile, steps, 1)

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


