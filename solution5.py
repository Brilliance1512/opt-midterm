import numpy as np
from numpy.linalg import solve

def f(x):
    return 25*x[0]**2 + 22*x[0]*x[1] + 7*x[1]**2 - 4*x[0] - 12*x[1]

def h(x):
    return 14*x[0]**2 - 36*x[0]*x[1] + 28*x[1]**2 - 9

def deltaH(x):
    return np.array([[28*x[0] - 36*x[1]], [-36*x[0] + 56*x[1]]])

def deltaF(x):
    return np.array([[50*x[0] + 22*x[1] - 4], [22*x[0] + 14*x[1] - 12]])

def D(lambd):
    return np.array([[50 + lambd*28, 22 - lambd*36], [22 - lambd*36, 14 - lambd*56]])


x = [3, 3]
lambd = 3
tol = 0.001
k = 0
pLen = tol


while True:
    gk = deltaF(x)
    Ak = deltaH(x)
    Dk = D(lambd)
    r = h(x)
    SolveMatrix = np.array([[Dk[0][0], Dk[0][1], Ak[0][0]], [Dk[1][0], Dk[1][1], Ak[1][0]], [Ak[0][0], Ak[1][0], 0]])
    equals = np.array([-gk[0][0], -gk[1][0], -r])
    solutions = solve(SolveMatrix, equals)
    pLen = max(abs(solutions[0]), abs(solutions[1]))
    if pLen > tol:
        x = x + np.array([solutions[0], solutions[1]])
        lambd = solutions[2]
        k += 1
    else:
        break
    
print('Number of iterations: ' + str(k))
print('Lambda: ' + str(lambd))
print('x: ({}, {})'.format(x[0], x[1]))
print('f(x): {}'.format(f(x)))
