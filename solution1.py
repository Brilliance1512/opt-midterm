import numpy as np

def f(x):
    return 6*x[0] + 9*x[1] + 6*x[2] + 9*x[3] + 12*x[4] + 14*x[5] + 7*x[6] + 11*x[7]

x = np.array([7, 3, 3, 4, 8, 3, 8, 8])
coefs = np.array([6, 9, 6, 9, 12, 12, 7, 11])

A = np.array([[2, 3, -9, 1, 10, -8, 9, 1],
              [-5, -2, -2, -3, 1, 7, -7, -9],
              [-5, -3, -4, 2, -1, -9, 7, 9],
              [3, -6, -4, 9, -9, 8, -7, -8]])

tol = 0.001
k = 0
r = 0.002
while r > tol:
    D = np.diag(x**2)
    S = A @ D
    H = S @ A.T
    h = -S @ coefs
    lambd = np.linalg.solve(H, h)
    delta = coefs + A.T @ lambd
    p = D @ delta
    r = np.sum(D @ delta**2)**0.5
    x = x - p/r
    k += 1

print('Number of iterations: ' + str(k))
print('x: {}'.format(x))
print('f(x): {}'.format(f(x)))
