import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from main import add_some, topple, create_exchange

# Parameters
threshold = 4
n = 100
size = n**2
steps = 1000

exchange = create_exchange(threshold)
sandpile = 3*np.ones(size)

# Configure the plot
fig, ax = plt.subplots()
im = ax.imshow(sandpile.reshape((n, n)), cmap='Greys', vmin=0, vmax=threshold, interpolation='nearest', origin='lower')


def update(frame):
    global sandpile
    sandpile = add_some(sandpile, 1)
    sandpile = topple(sandpile, exchange)
    im.set_array(sandpile.reshape((n, n)))
    return [im]


ani = FuncAnimation(fig, update, frames=steps, interval=1, blit=True, repeat=False)

plt.show()
