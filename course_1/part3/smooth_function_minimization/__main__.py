from math import sin, exp
from scipy.optimize import minimize


def f(x):
    return sin(x / 5) * exp(x / 10) + 5 * exp(-x / 2)


if __name__ == "__main__":
    print(minimize(f, [2]))
    print(minimize(f, [30], method="BFGS"))
