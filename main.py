# from functools import reduce
# from itertools import product
from math import sin, prod, sqrt
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la


def fun(x):
    return sin(x)


def custom_interpolation(points):
    x, y = zip(*points)
    _A = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            _A[i][j] = x[i] ** j
    coefficients = la.solve(_A, y)
    return lambda _x: sum(coeff * (_x ** i) for coeff, i in zip(coefficients, range(len(coefficients))))


def l(j):
    return lambda _xs: \
        lambda x: prod([((x - _x) / (_xs[j] - _x)) for _x, i in zip(_xs, range(len(_xs))) if i != j])


def uppercase_l(points):
    return lambda _x: sum(y * l(i)(list(zip(*points))[0])(_x) for x, y, i in
                          zip(*(list(zip(*points)) + list([tuple(range(len(points)))]))))


def divided_diff(x, y, j):
    if j == 0:
        return y[0]
    # n = np.shape(y)[0]
    helper = np.zeros([j + 1, j + 1])
    helper[::, 0] = y[:j + 1:]
    for k in range(1, j + 1):
        for i in range(j + 1 - k):
            helper[i][k] = (helper[i + 1][k - 1] - helper[i][k - 1]) / (x[i + k] - x[i])
    return helper[0][j]


def n(j):
    return lambda _xs: \
        lambda x: 1 if j == 0 else prod([(x - _x) for _x, i in zip(_xs, range(j))])


def uppercase_n(points):
    x, y = zip(*points)
    return lambda _x: sum([divided_diff(x, y, i) * n(i)(x)(_x) for i in
                           range(len(points))])


def cubic_interp1d(points):
    x, y = zip(*points)
    x = np.asfarray(x)
    y = np.asfarray(y)

    size = len(x)

    xdiff = np.diff(x)
    ydiff = np.diff(y)

    Li = np.empty(size)
    Li_1 = np.empty(size - 1)
    z = np.empty(size)

    # [L][y] = [B]
    Li[0] = sqrt(2 * xdiff[0])
    Li_1[0] = 0.0
    B0 = 0.0
    z[0] = B0 / Li[0]

    for i in range(1, size - 1, 1):
        Li_1[i] = xdiff[i - 1] / Li[i - 1]
        Li[i] = sqrt(2 * (xdiff[i - 1] + xdiff[i]) - Li_1[i - 1] * Li_1[i - 1])
        Bi = 6 * (ydiff[i] / xdiff[i] - ydiff[i - 1] / xdiff[i - 1])
        z[i] = (Bi - Li_1[i - 1] * z[i - 1]) / Li[i]

    i = size - 1
    Li_1[i - 1] = xdiff[-1] / Li[i - 1]
    Li[i] = sqrt(2 * xdiff[-1] - Li_1[i - 1] * Li_1[i - 1])
    Bi = 0.0
    z[i] = (Bi - Li_1[i - 1] * z[i - 1]) / Li[i]

    # [L.T][x] = [y]
    z[i] = z[i] / Li[i]
    for i in range(size - 2, -1, -1):
        z[i] = (z[i] - Li_1[i - 1] * z[i + 1]) / Li[i]

    def helper(x0):
        index = x.searchsorted(x0)
        np.clip(index, 1, size - 1, index)

        xi1, xi0 = x[index], x[index - 1]
        yi1, yi0 = y[index], y[index - 1]
        zi1, zi0 = z[index], z[index - 1]
        hi1 = xi1 - xi0

        f0 = zi0 / (6 * hi1) * (xi1 - x0) ** 3 + \
             zi1 / (6 * hi1) * (x0 - xi0) ** 3 + \
             (yi1 / hi1 - zi1 * hi1 / 6) * (x0 - xi0) + \
             (yi0 / hi1 - zi0 * hi1 / 6) * (xi1 - x0)
        return f0

    return helper


a: Final = 0
b: Final = 5
N: Final = 5
xs = list(range(a, b + 1, int((b - a) / N)))
ys = list(map(lambda x: fun(x), xs))
pairs = list(zip(xs, ys))
testvar = np.linspace(min(0, a - 1), b + 1, 100)

fig, ((ax, custom), (lagrange, newton), (cubic_spline, _)) = plt.subplots(3, 2)

ax.plot(testvar, custom_interpolation(pairs)(testvar), label='custom')
ax.plot(testvar, uppercase_l(pairs)(testvar), label='Lagrange')
ax.plot(testvar, uppercase_n(pairs)(testvar), label='Newton')
ax.plot(xs, ys, 'ro')

custom.plot(xs, ys, 'ro')
lagrange.plot(xs, ys, 'ro')
newton.plot(xs, ys, 'ro')
cubic_spline.plot(xs, ys, 'ro')

custom.plot(testvar, custom_interpolation(pairs)(testvar), label='custom')
lagrange.plot(testvar, uppercase_l(pairs)(testvar), label='Lagrange')
newton.plot(testvar, uppercase_n(pairs)(testvar), label='Newton')
cubic_spline.plot(testvar, cubic_interp1d(pairs)(testvar), label='Cubic spline')

ax.legend()
custom.legend()
lagrange.legend()
newton.legend()
cubic_spline.legend()

plt.show()
