import numpy as np
from scipy.signal import convolve2d, welch
import matplotlib.pyplot as plt
import powerlaw
import time
from scipy.optimize import curve_fit


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

        # Initialise grid with a uniform cover of sand
        self.n = n
        self.m = m
        self.grid = cover * np.ones((n, m), dtype=np.int32)

        # Initialise storage of data
        self.topples = []
        self.avl_topples = []
        self.avl_areas = []
        self.avl_time = []

    def add_one(self):
        # Generates a random index and adds unit of sand to cell with this index

        rand_index = np.random.randint(self.n), np.random.randint(self.m)
        self.grid[rand_index] += 1

    def stabilise(self):
        # Executes the stabilising operator on sandpile configuration
        # returns: the record of cells toppled and the duration of avalanche

        # Initialise grid to record cells that topple over avalanche
        topple_record = np.zeros((self.n, self.m), dtype=np.int32)
        # Set the clock
        avl_time = 0
        while (self.grid >= 4).any():
            avl_time += 1
            unstable = self.grid >= 4
            self.topples.append(np.sum(unstable))
            topple_record += unstable

            exchange = convolve2d(unstable, self.kernel, mode='same')
            self.grid += exchange
        return topple_record, avl_time

    def add_and_topple(self, steps=100000):
        for i in range(steps):
            self.add_one()
            topple_record, avl_time = self.stabilise()

            # Record the total cells affected by avalanche
            topple_map = topple_record > 0

            # Record data for each time step
            self.avl_topples.append(np.sum(topple_record))
            self.avl_areas.append(np.sum(topple_map))
            self.avl_time.append(avl_time)


# Set up grid into probable critical state
pile = Sandpile(n=20, cover=30)
pile.stabilise()


# Simulate sandpile dynamics over a number of time steps
pile.add_and_topple(steps=100000)

# Extract data on global dynamics
avl_topples, avl_areas, avl_time = np.array(pile.avl_topples), np.array(pile.avl_areas), np.array(pile.avl_time)


# Filter trivial data
data = [avl_topples[avl_topples > 0], avl_areas[avl_areas > 0], avl_time[avl_time > 0]]

# Configure the plot
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
titles = ["Number of topples", "Area of avalanche", "Avalanche duration"]

# Fit data to power law
for i in range(3):
    fit = powerlaw.Fit(data[i], xmin=min(data[i]), xmax=1000, discrete=True)
    fit.power_law.plot_pdf(
        color='gray',
        linestyle='--',
        linewidth=0.8,
        ax=ax[i],
        label=r'$t^{{-\alpha}}: \alpha$={:.2f}'.format(fit.alpha))
    fit.plot_pdf(
        color='k',
        ax=ax[i],
        linewidth=0.8,
        label="PDF")

    ax[i].set_xlabel(titles[i])
    ax[i].legend()

plt.savefig('power_laws_20.pdf')

# Pink noise calculation #

freq, power = welch(pile.topples, nperseg=2048, fs=2048)

# Consider frequencies greater than 10 units of time
idx = np.argmax(freq >= 10)
freq = freq[idx:]
power = power[idx:]

# Take the logarithm of `freq` and `power`
log_f = np.log10(freq)
log_p = np.log10(power)


# Define the power law function to fit
def power_law_func(t, a, b):
    return a * t + b
# Fit the power law function to the data using curve_fit() from scipy.optimize
coeffs, _ = curve_fit(power_law_func, log_f, log_p)

# Extract the fitted coefficients
a, b = coeffs
# Generate the fitted power law curve
fit = 10 ** (a * log_f + b)

# Plot the distribution and fit

fig, ax = plt.subplots()
plt.loglog(freq, power, 'grey', label='power spectral density', alpha=0.6)
plt.loglog(freq, fit, 'k', linestyle='--', label=r'$t^{{-\alpha}}: \alpha$={:.2f}'.format(a))
plt.legend()
plt.savefig('pink_noise.pdf')
np.savetxt('coefficients_pink_noise_m_c.txt', coeffs)

# Fractal dimension #

# Create a dictionary recording the cells at each height level
levels = {0: None, 1: None, 2: None, 3: None}
for i in range(4):
    mask = (pile.grid == i)
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
        bin_count, edges = np.histogramdd(cells, bins=(np.arange(0, pile.n, scale), np.arange(0, pile.n, scale)))
        # Find number of bins with non-zero cells
        counts.append(np.sum(bin_count > 0))
    # linear fit, polynomial of degree 1
    coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
    return scales, counts, coeffs


fig, axs = plt.subplots(1, 4)

for i, ax in enumerate(axs.flatten()):
    if i < 4:
        ax.imshow(levels[i], cmap='binary', interpolation='nearest', origin='lower')
        ax.set_title(f"Height level {i}")
        ax.axis('off')

plt.savefig('grid_fractal.pdf')

fig, axs = plt.subplots(1, 4)

for i, ax in axs:
    scales, counts, coeffs = box_count(levels[i - 4])
    print("The Hausdorff dimension is", -coeffs[0])
    ax.plot(np.log(scales), np.log(counts), 'o', mfc='none')
    ax.plot(np.log(scales), np.polyval(coeffs, np.log(scales)))
    ax.set_xlabel('log $\epsilon$')
    ax.set_ylabel('log N')
    ax.set_title(f"Level {i - 4}")

plt.savefig('box_counting_regression.pdf')
np.savetxt('hausdorff_dim_0_to_3.txt', coeffs)
