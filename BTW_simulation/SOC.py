import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import powerlaw


class Sandpile:
    # Abelian Sandpile
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.int32)

    def __init__(self, n, m=None, cover=10):
        # Initialises the attributes.

        # n: number of rows
        # m: number of columns
        # cover: the starting value for all cells

        m = n if m is None else m

        self.grid = cover * np.ones((n, m), dtype=np.int32)

    def add_one(self):
        grid = self.grid
        n, m = grid.shape
        rand_index = np.random.randint(n), np.random.randint(m)
        grid[rand_index] += 1

    def topple(self):
        # Executes the stabilising operator on sandpile configuration
        # returns: the record of cells toppled and the duration of avalanche

        grid = self.grid
        n, m = grid.shape
        topple_record = np.zeros((n, m), dtype=np.int32)
        avl_time = 0
        while (grid >= 4).any():
            unstable = grid >= 4
            avl_time += 1
            topple_record += unstable
            exchange = convolve2d(unstable, self.kernel, mode='same')
            grid += exchange
        return topple_record, avl_time

    def add_and_topple(self):
        # Adds unit of sand to sandpile and topples until configuration stable
        # returns: duration of avalanche, total toppled, avalanche area

        self.add_one()
        topple_record, avl_time = self.topple()
        topple_map = topple_record > 0
        return avl_time, np.sum(topple_record), np.sum(topple_map)


pile = Sandpile(n=100, cover=30)
pile.topple()

stats = [pile.add_and_topple() for _ in range(10000)]

durations, topples, areas = np.transpose(stats)

stats = [durations[durations > 0], topples[topples > 0], areas[areas > 0]]
print(durations, topples, areas)
print(durations.size, topples.size, areas.size)


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i in range(3):
    fit = powerlaw.Fit(stats[i], discrete=True)
    print('alpha:', fit.alpha, 'sigma:', fit.sigma)

    fit.power_law.plot_pdf(color='gray', linestyle='--', ax=axes[i])
    fit.plot_pdf(color='k', ax=axes[i])
    axes[i].set_xlabel('Topple size')
    axes[i].set_ylabel('Frequency')
    axes[i].set_xscale('log')
    axes[i].set_yscale('log')
    axes[i].set_title(f'Subplot {i+1}')

plt.show()
