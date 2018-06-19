from math import sin, exp
from scipy.optimize import differential_evolution


def f(x):
    return sin(x / 5) * exp(x / 10) + 5 * exp(-x / 2)


if __name__ == "__main__":
    print(differential_evolution(f, [(1, 30)]))
