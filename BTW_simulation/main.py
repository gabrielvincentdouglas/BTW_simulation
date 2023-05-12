import numpy as np
from scipy.signal import convolve2d
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Parameters
threshold = 4
n = 50


def topple(sandpile):
    while (sandpile >= 4).any():
        unstable = sandpile >= threshold
        exchange = convolve2d(unstable, kernel, mode='same')
        sandpile += exchange
    return sandpile


def add_some(sandpile, amount):
    new_sand = np.zeros((n, n))
    rand_index = np.random.randint(n), np.random.randint(n)
    new_sand[rand_index] += amount
    sandpile += new_sand.astype(int)
    return sandpile.astype(int)


# Simulate sandpile evolution
def add_and_topple(sandpile, steps, amount):
    for i in range(steps):
        sandpile = add_some(sandpile, amount)
        sandpile = topple(sandpile)
    return sandpile


kernel = np.array([[0,  1, 0],
                   [1, -4, 1],
                   [0,  1, 0]])

times = []

for n in (20, 50, 100, 150):
    start_time = time.time()

    sandpile = add_and_topple(5 * np.ones((n, n)), steps=1000, amount=1)

    end_time = time.time()

    duration = end_time - start_time

    times.append(duration)

np.savetxt('times.csv', times, delimiter=',', newline=';')


def power_law_func(t, a, b, c):
    return a * t**2 + b*t + c


x = np.array([20**2, 50**2, 100**2, 150**2])
# Fit the power law function to the data using curve_fit() from scipy.optimize
coeffs, _ = curve_fit(power_law_func, x, times)

np.savetxt('timing_coeffs.csv', coeffs, delimiter=',', newline=';')
# Extract the fitted coefficients
a, b, c = coeffs

x_finer = np.linspace(x.min(), x.max(), 100)

# Compute the quadratic fit using the fitted coefficients and the new x values
fit = a * x_finer**2 + b * x_finer + c

fig, ax = plt.subplots()

plt.plot(x, times, 'o', label='Data')
plt.plot(x_finer, fit, label='$O(n^2)$')

plt.xlabel("Number of cells")
plt.ylabel("Computation time")
plt.legend()
plt.savefig("timing_convolution.pdf")

plt.show()
