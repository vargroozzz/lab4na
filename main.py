from functools import reduce
from itertools import product
from math import sin, prod
from typing import Final

import matplotlib.pyplot as plt
import numpy as np


def fun(x):
    return sin(x)


def l(j):
    return lambda _xs: \
        lambda x: prod([((x - _x) / (_xs[j] - _x)) for _x, i in zip(_xs, range(len(_xs))) if i != j])


def uppercase_l(points):
    return lambda _x: sum(y * l(i)(list(zip(*points))[0])(_x) for x, y, i in
                          zip(*(list(zip(*points)) + list([tuple(range(len(points)))]))))


def divided_diff(x, y, j):
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


# def uppercase_n(points):
#     return lambda _x: sum([divided_diff(x, y, i) * n(i)(points[0])(_x) for x, y, i in
#                            zip(*(list(zip(*points)) + list([tuple(range(len(points)))])))])
def uppercase_n(points):
    return lambda _x: sum([divided_diff(points[0], points[1], i) * n(i)(points[0])(_x) for x, y, i in
                           zip(*(list(zip(*points)) + list([tuple(range(len(points)))])))])


a: Final = 0
b: Final = 5
N: Final = 5
xs = list(range(a, b + 1, int((b - a) / N)))
ys = list(map(lambda x: fun(x), xs))
pairs = list(zip(xs, ys))
testvar = np.linspace(min(0, a - 1), b + 1, 100)
fig, ax = plt.subplots()
ax.plot(xs, ys, 'ro')
ax.plot(testvar, uppercase_l(pairs)(testvar))
ax.plot(testvar, uppercase_n(pairs)(testvar))
plt.show()
