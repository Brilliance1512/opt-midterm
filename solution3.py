import numpy as np
from scipy.optimize import approx_fprime, minimize_scalar

n = 10
tol = 0.01
x = np.zeros(n)


def f(x):
    res = 0
    for i in range(0, n-1):
        res += (x[i+1] - x[i] + 1 - (x[i])**2)**2
    return res*20

g = approx_fprime(x, f, 1e-6)
d = -g
k = 1
while np.linalg.norm(g) >= tol:
    rk = minimize_scalar(lambda t: f(x + d * t)).x
    x = x + rk*d
    gpr = g
    g = approx_fprime(x, f, 1e-6)
    beta = (g.T @ g)/(gpr.T @ gpr) if k % len(x) == 0 else 0
    d = -g + beta*d
    k += 1

print('Number of iterations: ' + str(k))
print('x: {}'.format(x))
print('f(x): {}'.format(f(x)))
