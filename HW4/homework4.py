import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
import matplotlib.colors as mcolors

## Armijo Line Search

gamma_0 = 1
alpha = 1e-4
beta = 0.5
max_iter = 100
x = np.array([-4, -2])

def f(x):
    return 0.26 * (x[0]**2 + x[1]**2) - 0.46 * x[0] * x[1]

def grad_f(x):
    return np.array([0.52 * x[0] - 0.46 * x[1], 0.52 * x[1] - 0.46 * x[0]])

def armijo_line_search(x, gamma_0, alpha, beta, max_iter):
    gamma = gamma_0
    for i in range(max_iter):
        if f(x - gamma * grad_f(x)) <= f(x) - alpha * gamma * np.linalg.norm(grad_f(x))**2:
            break
        gamma *= beta
    return gamma

def plot_trajectory(x, gamma, max_iter):
    x_trajectory = np.zeros((max_iter + 1, 2))
    x_trajectory[0] = x
    for i in range(max_iter):
        x = x - gamma * grad_f(x)
        x_trajectory[i + 1] = x
    return x_trajectory

gamma = armijo_line_search(x, gamma_0, alpha, beta, max_iter)
x_trajectory = plot_trajectory(x, gamma, max_iter)

X = np.linspace(-5, 5, 100)
Y = np.linspace(-5, 5, 100)

X, Y = np.meshgrid(X, Y)
Z = f([X, Y])

plt.figure()
plt.contourf(X, Y, Z, cmap='Blues_r', alpha=0.8, norm = mcolors.LogNorm())
plt.plot(x_trajectory[:, 0], x_trajectory[:, 1], '-', color='purple')
plt.scatter(x_trajectory[:, 0][0], x_trajectory[:, 1][0], c='r', alpha=1.)
plt.scatter(x_trajectory[:, 0][-1], x_trajectory[:, 1][-1], c='r', alpha=1.)

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Armijo Line Search')
plt.grid()
plt.show()



##### iLQR for diff drive robot

T = 2 * np.pi






