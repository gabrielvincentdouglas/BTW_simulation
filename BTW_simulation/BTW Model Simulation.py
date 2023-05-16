#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.signal import convolve2d


# In[52]:


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

    def stabilise(self, display=False):
        # Executes the stabilising operator on sandpile configuration
        # returns: the record of cells toppled and the duration of avalanche

        # Initialise grid to record cells that topple over avalanche
        topple_record = np.zeros((self.n, self.m), dtype=np.int32)
        # Set the clock
        avl_time = 0
        
        # Display the grid if display set to true
        if display:
                print("Inital Grid")
                print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in self.grid]))
        
        while True:
            unstable = self.grid >= 4
            num_unstable = np.count_nonzero(unstable)

            if num_unstable == 0:
                break

            avl_time += 1
            topple_record += unstable
            self.topples.append(np.sum(unstable))

            exchange = convolve2d(unstable, self.kernel, mode='same')
            self.grid += exchange
            
            if display:
                print("Step",avl_time)
                print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in self.grid]))
                
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
            


# In[53]:


pile = Sandpile(n=5, m=4, cover=0)
pile.grid[0,2] = 4
pile.grid[1,2] = 3
pile.grid[1,3] = 3


pile.stabilise(display=True);


# In[23]:


# Enable interactive plot
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[33]:


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# In[40]:


sandpile = Sandpile(30, cover=0)

# Create a figure and axis object for the animation
fig, ax = plt.subplots()
im = ax.imshow(sandpile.grid, cmap='Greys', vmin=0, vmax=3, interpolation='nearest')
fig.colorbar(im, ax=ax)

# Define the animation function that adds one grain of sand, stabilises the sandpile,
# and updates the plot with the new grid
def animate(i):
    sandpile.add_and_topple(steps=1)
    
    # Update the plot with the new grid
    im.set_data(sandpile.grid)
    ax.set_title(f"Step {i+1}")
    
    return im,

# Create the animation
anim = FuncAnimation(
    fig, animate, frames=1000, blit=True, interval=5, repeat=False)

# Save the animation as gif
writer = PillowWriter(fps=50)
anim.save('sandpile.gif', writer=writer)


# Let's simulate a larger grid over 100,000 time steps

# In[54]:


# Set up grid
pile = Sandpile(n=100, cover=0)

# Simulate sandpile dynamics over a number of time steps
pile.add_and_topple(steps=100000)


# Power laws

# In[45]:


import powerlaw


# In[57]:


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

# save the figure to a PDF file
fig.savefig('power_laws.pdf', dpi=300, bbox_inches='tight')


# Pink Noise Calculation

# In[48]:


from scipy.optimize import curve_fit
from scipy.signal import welch


# In[59]:


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
# Fit the power law function to the data 
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


# Investigate fractal geometry

# In[69]:


pile2 = Sandpile(n=100, cover=20)
pile2.stabilise()

# Create a dictionary recording the cells at each height level
levels = {0: None, 1: None, 2: None, 3: None}
for i in range(4):
    mask = (pile2.grid == i)
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
    ax.imshow(levels[i], cmap='binary', interpolation='nearest', origin='lower')
    ax.set_title(f"Height level {i}")
    ax.axis('off')

plt.savefig('grid_fractal.pdf')
plt.show()

H_dimension = []

for i in range(4):
    scales, counts, coeffs = box_count(levels[i])
    H_dimension.append(-coeffs[0])
    print(f"The Hausdorff dimension for height level {i}:", -coeffs[0])
    
np.savetxt('H_dimension.txt', H_dimension)


# This is a method that topples the sandpile such that sand is distributed randomly with weighting relative to the height difference with neighbours. It provides the option to include a slope in the grid.
#         

# In[101]:


import random


# In[152]:


class Aug_Sandpile:

    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.int32)

    def __init__(self, n, cover=10):
        # Initialises the attributes.

        # n: length of grid
        # cover: the starting value for all cells

        self.n = n
        self.size = n**2

        self.grid = cover * np.ones(self.size, dtype=np.int32)

        self.avl_topples = []
        self.avl_areas = []
        self.avl_time = []

    def add_one(self):
        # Generates a random index and adds unit of sand to cell with this index

        rand_index = np.random.randint(self.size)
        self.grid[rand_index] += 1
    
    def random_sand(self, weight):
        sand_fall = [0, 0, 0, 0]
        # Randomly select with repetition which adjacent cells collect toppled sand
        indices = random.choices(range(4), weight, k=4)
        for i in indices:
            sand_fall[i] += 1
        return sand_fall


    def stabilise(self, slope, display=False):

        topple_record = np.zeros(self.size)   
        avl_time = 0    
        
        # Display the grid if display set to true
        if display:
            grid=self.grid.reshape((self.n, self.n))
            print("Inital Grid")
            print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in grid]))
        
        while True:
            unstable = self.grid >= 4
            num_unstable = np.count_nonzero(unstable)

            if num_unstable == 0:
                break

            avl_time += 1   
            topple_record += unstable
            
            n = self.n
            size = self.size
            sandpile = self.grid
            
            for i in range(size):
                if sandpile[i] >= 4:
                    if i == 0:
                        diff = sandpile[i] - np.array([sandpile[i + n], sandpile[i + 1], 0, 0])
                        weight = np.maximum(diff, 0) + slope
                        sand_fall = self.random_sand(weight)
                        sandpile[i + n] += sand_fall[0]
                        sandpile[i + 1] += sand_fall[1]
                    elif i == n - 1:
                        diff = sandpile[i] - np.array([sandpile[i + n], 0, 0, sandpile[i - 1]])
                        weight = np.maximum(diff, 0) + slope
                        sand_fall = self.random_sand(weight)
                        sandpile[i + n] += sand_fall[0]
                        sandpile[i - 1] += sand_fall[3]
                    elif i == size - n:
                        diff = sandpile[i] - np.array([0, sandpile[i + 1], sandpile[i - n], 0])
                        weight = np.maximum(diff, 0) + slope
                        sand_fall = self.random_sand(weight)
                        sandpile[i + 1] += sand_fall[1]
                        sandpile[i - n] += sand_fall[2]
                    elif i == size - 1:
                        diff = sandpile[i] - np.array([0, 0, sandpile[i - n], sandpile[i - 1]])
                        weight = np.maximum(diff, 0) + slope
                        sand_fall = self.random_sand(weight)
                        sandpile[i - n] += sand_fall[2]
                        sandpile[i - 1] += sand_fall[3]
                    elif i in range(n):
                        diff = sandpile[i] - np.array([sandpile[i + n], sandpile[i + 1], 0, sandpile[i - 1]])
                        weight = np.maximum(diff, 0) + slope
                        sand_fall = self.random_sand(weight)
                        sandpile[i + n] += sand_fall[0]
                        sandpile[i + 1] += sand_fall[1]
                        sandpile[i - 1] += sand_fall[3]
                    elif i % n == 0:
                        diff = sandpile[i] - np.array([sandpile[i + n], sandpile[i + 1], sandpile[i - n], 0])
                        weight = np.maximum(diff, 0) + slope
                        sand_fall = self.random_sand(weight)
                        sandpile[i + n] += sand_fall[0]
                        sandpile[i + 1] += sand_fall[1]
                        sandpile[i - n] += sand_fall[2]
                    elif i % n == n - 1:
                        diff = sandpile[i] - np.array([sandpile[i + n], 0, sandpile[i - n], sandpile[i - 1]])
                        weight = np.maximum(diff, 0) + slope
                        sand_fall = self.random_sand(weight)
                        sandpile[i + n] += sand_fall[0]
                        sandpile[i - n] += sand_fall[2]
                        sandpile[i - 1] += sand_fall[3]
                    elif i in range(size - n, size):
                        diff = sandpile[i] - np.array([0, sandpile[i + 1], sandpile[i - n], sandpile[i - 1]])
                        weight = np.maximum(diff, 0) + slope
                        sand_fall = self.random_sand(weight)
                        sandpile[i + 1] += sand_fall[1]
                        sandpile[i - n] += sand_fall[2]
                        sandpile[i - 1] += sand_fall[3]
                    else:
                        diff = sandpile[i] - np.array([sandpile[i + n], sandpile[i + 1], sandpile[i - n], sandpile[i - 1]])
                        weight = np.maximum(diff, 0) + slope
                        sand_fall = self.random_sand(weight)
                        sandpile[i + n] += sand_fall[0]
                        sandpile[i + 1] += sand_fall[1]
                        sandpile[i - n] += sand_fall[2]
                        sandpile[i - 1] += sand_fall[3]
                    sandpile[i] -= 4
                    if display:
                        grid =self.grid.reshape((self.n, self.n))
                        print("Step", avl_time)
                        print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in grid]))
                
        return topple_record, avl_time


    def add_and_stabilise(self, steps=100000, display=False):
        
        ans = input("Do you want to include a slope (southwards) on the grid? (yes/no)" )
        if ans.lower() == "yes":
            slope = [10, 1, 1, 1]
        else:
            slope = [1, 1, 1, 1]
            
            
        for i in range(steps):
            self.add_one()
            topple_record, avl_time = self.stabilise(slope, display)

            # Record the total cells affected by avalanche
            topple_map = topple_record > 0

            # Record data for each time step
            self.avl_topples.append(np.sum(topple_record))
            self.avl_areas.append(np.sum(topple_map))
            self.avl_time.append(avl_time)


# In[153]:


pile3 = Aug_Sandpile(n=5, cover=0)
pile3.grid[13] = 4
pile3.grid[18] = 3
pile3.grid[23] = 3


pile3.add_and_stabilise(steps= 1, display=True);


# In[156]:


pile4 = Aug_Sandpile(n=100, cover=0)

pile4.add_and_stabilise(steps=100000)


# In[157]:


# Extract data on global dynamics
avl_topples, avl_areas, avl_time = np.array(pile4.avl_topples), np.array(pile4.avl_areas), np.array(pile4.avl_time)


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

# save the figure to a PDF file
fig.savefig('power_laws_aug.pdf', dpi=300, bbox_inches='tight')


# In[ ]:




