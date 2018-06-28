from math import sin, exp
from scipy.optimize import minimize, differential_evolution

def f(x):
    return sin(x / 5) * exp(x / 10) + 5 * exp(-x / 2)


def h(x):
    return int(f(x))


if __name__ == "__main__":
    print(minimize(h, [30], method="BFGS"))
    print(differential_evolution(h, [(1, 30)]))
