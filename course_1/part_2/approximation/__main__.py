import numpy as np
from scipy import linalg
from math import sin, exp


def f(x):
    return sin(x / 5) * exp(x / 10) + 5 * exp(-x / 2)


if __name__ == "__main__":
    A = np.array([[1, 1, 1**2, 1**3], [1, 4, 4**2, 4**3], [1, 10, 10**2, 10**3], [1, 15, 15**2, 15**3]])
    b = np.array([f(1), f(4), f(10), f(15)])
    x = linalg.solve(A, b)
    print(x)
    print(x[0] + x[1] * 1 + x[2] * 1**2 + x[3] * 1**3, f(1))
    print(x[0] + x[1] * 4 + x[2] * 4**2 + x[3] * 4**3, f(4))
    print(x[0] + x[1] * 10 + x[2] * 10**2 + x[3] * 10**3, f(10))
    print(x[0] + x[1] * 15 + x[2] * 15**2 + x[3] * 15**3, f(15))
