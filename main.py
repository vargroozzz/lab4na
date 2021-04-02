from functools import reduce
from itertools import product
from math import sin, prod
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


a: Final = 0
b: Final = 5
N: Final = 5
xs = list(range(a, b + 1, int((b - a) / N)))
ys = list(map(lambda x: fun(x), xs))
pairs = list(zip(xs, ys))
testvar = np.linspace(min(0, a - 1), b + 1, 100)
# fig, ax = plt.subplots()
# ax.plot(xs, ys, 'ro')
# ax.plot(testvar, custom_interpolation(pairs)(testvar), label='custom')
# ax.plot(testvar, uppercase_l(pairs)(testvar), label='Lagrange')
# ax.plot(testvar, uppercase_n(pairs)(testvar), label='Newton')
# ax.legend()
fig, ((ax, custom), (lagrange, newton)) = plt.subplots(2, 2)
ax.plot(testvar, custom_interpolation(pairs)(testvar), label='custom')
ax.plot(testvar, uppercase_l(pairs)(testvar), label='Lagrange')
ax.plot(testvar, uppercase_n(pairs)(testvar), label='Newton')
ax.plot(xs, ys, 'ro')

custom.plot(xs, ys, 'ro')
lagrange.plot(xs, ys, 'ro')
newton.plot(xs, ys, 'ro')

custom.plot(testvar, custom_interpolation(pairs)(testvar), label='custom')
lagrange.plot(testvar, uppercase_l(pairs)(testvar), label='Lagrange')
newton.plot(testvar, uppercase_n(pairs)(testvar), label='Newton')

ax.legend()
custom.legend()
lagrange.legend()
newton.legend()

plt.show()
