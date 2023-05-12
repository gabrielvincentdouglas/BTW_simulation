import numpy as np
import matplotlib.pyplot as plt

times = [1.651349067687988281e-01,1.810950040817260742e+00,1.762247896194458008e+01,8.196909117698669434e+01]
x = np.array([20**2, 50**2, 100**2, 150**2])
x_finer = np.linspace(0, x.max(), 100)

fit = 1.76e-07*x_finer**2

fig, ax = plt.subplots()

plt.plot(x, times, 'o', label='Data')
plt.plot(x_finer, fit, label='$C n^2$')

plt.xlabel("Number of cells")
plt.ylabel("Computation time")
plt.legend()
plt.savefig("timing_convolution.pdf")
plt.show()