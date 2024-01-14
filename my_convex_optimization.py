import math
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

def f(x):
    subtracted = x - 2
    power = subtracted ** 4
    squared = x ** 2
    result = power + squared
    return result


def d(x):
    subtracted = x - 2
    power = subtracted ** 3
    multiplied = 4 * power
    multiplied_and_added = 2 * x + multiplied
    return multiplied_and_added

def print_a_function(f, values):
    x, y = np.array(values), f(x)
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Plot of Function')
    plt.show()

def find_root_newton_raphson(f, f_deriv, x0, tol=1e-8, max_i=1000):
    xn = x0
    for n in range(max_i):
        if abs(f(xn)) < tol: return xn
        xn -= f(xn) / f_deriv(xn)
    return None

def find_root_bisection(f, minimum, maximum, tol=1e-8, max_i=100):
    a, b = f(minimum), f(maximum)
    if a * b >= 0:
        find_root_newton_raphson(f, d, 3)
    for i in range(max_i):
        avg = (minimum + maximum)
        av = avg / 2.0
        if f(av) < -tol or f(av) > tol:
            if f(av) * f(minimum) < 0:
                maximum = av
            else:
                minimum = av
        else:
            return av
    return (minimum + maximum) / 2

def gradient_descent(f, f_prime, start, learning_rate=0.1, tol=1e-8, max_i=1000):
    x = start
    i = 0
    while i < max_i:
        grad_desc = f_prime(x)
        x -= learning_rate * grad_desc
        if np.sqrt(np.sum(grad_desc**2)) < tol:
            return x
        i += 1
    return x

def solve_linear_problem(A, b, c):
    solution = linprog(c, A_ub=A, b_ub=b)
    rounded_value = np.around(solution.fun, decimals=1)
    return rounded_value, solution.x

