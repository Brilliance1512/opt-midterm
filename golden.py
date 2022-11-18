import math
def myf(x):
    return 1.4*math.sin(x)*math.cos(x) + (0.25*(x - 5)**2)*math.log(x) - 1.2*x

def golden(start, end, e):
    xs, xe, k = start, end, 0
    gamma = (math.sqrt(5) - 1)/2
    ys = xs + (1 - gamma)*(xe - xs)
    ye = xs + gamma*(xe - xs)
    fs = myf(ys)
    fe = myf(ye)
    while xe - xs > e:
        if fs > fe:
            xs = ys
            ys = ye
            fs = fe
            ye = xs + gamma*(xe - xs)
            fe = myf(ye)
        else:
            xe = ye
            ye = ys
            fe = fs
            ys = xs + (1 - gamma)*(xe - xs)
            fs = myf(ys)
        k += 1
    return (xs, myf(xs), k)
min_x, min_y, tries = golden(1.5, 10, 0.01)
print('Minimum is attained at ({}, {}), calculated with {} iterations'.format(min_x, min_y, tries))
    
