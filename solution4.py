import numpy as np
H = np.identity(10)
k = 0
tol = 0.001
n = 10
x = [1.1, 0.1, 1.1, 0.1, 1.1, 0.1, 1.1, 0.1, 1.1, 0.1]

def f(x):
    res = 0
    for i in range(1, n//2 + 1):
        res += (x[2*i-2])**2 + 10*((x[2*i-2])**2 + (x[2*i-1])**2 - 1)**2
    return res

def grad(x):
    return np.array([40*x[0]*(x[0]**2 + x[1]**2 - 1) + 2*x[0],
                     40*x[1]*(x[0]**2 + x[1]**2 - 1),
                     40*x[2]*(x[2]**2 + x[3]**2 - 1) + 2*x[2],
                     40*x[3]*(x[2]**2 + x[3]**2 - 1),
                     40*x[4]*(x[4]**2 + x[5]**2 - 1) + 2*x[4],
                     40*x[5]*(x[4]**2 + x[5]**2 - 1),
                     40*x[6]*(x[6]**2 + x[7]**2 - 1) + 2*x[6],
                     40*x[7]*(x[6]**2 + x[7]**2 - 1),
                     40*x[8]*(x[8]**2 + x[9]**2 - 1) + 2*x[8],
                     40*x[9]*(x[8]**2 + x[9]**2 - 1)])
        
def wolfe(x, d):
    sigma1 = 0.3
    sigma3 = 0.7
    theta1 = 2
    theta2 = 0.5
    tauL = 0
    tauR = 0
    tau = 3
    w1 = False
    w2 = False
    while not (w1 and w2):
        w1 = f(x + tau*d) <= f(x) + tau*sigma1*(grad(x).T.dot(d))
        w2 = grad(x + tau*d).T.dot(d) >= sigma3*(grad(x).T.dot(d))
        if not w1:
            tauR = tau
            tau = (1 - theta2)*tauL + theta2*tauR
        else:
            if not w2:
                tauL = tau
            if tauR == 0:
                tau = theta1*tau
            else:
                tau = (1 - theta2)*tauL + theta2*tauR
    return tau

def DFP(deltak, pk, Hk):
    partOne = (deltak @ deltak.T)/(pk.T @ deltak)
    partTwo = (Hk @ pk @ pk.T * Hk)/(pk.T @ Hk @ pk)
    return partOne - partTwo

g = grad(x)
while np.linalg.norm(g) > tol:
    dk = -H.dot(g)
    rk = wolfe(x, dk)
    deltaX = rk*dk
    x = x + deltaX
    deltaGrad = grad(x) - g
    deltaH = DFP(deltaX, deltaGrad, H)
    H = H - deltaH
    g = grad(x)
    k += 1

print('Number of iterations: ' + str(k))
print('x: {}'.format(x))
print('f(x): {}'.format(f(x)))
