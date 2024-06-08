if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from mydezero import Variable
from mydezero import Function
from mydezero.utils import plot_dot_graph
import math
def sphere(x,y):
    z = x ** 2 + y ** 2
    return z

def mytans(x,y):
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return z

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx

def sin(x):
    return Sin()(x)

def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y

def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2)**2 + (x0 - 1) ** 2
    return y

def f(x):
    y = x ** 4 - 2 * x ** 2
    return y

def gx2(x):
    return 12 * x ** 2 - 4

x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i,x)

    y = f(x)
    print(y)
    x.cleargrad()
    y.backward()
    print(x.grad)
    x.data -= x.grad / gx2(x.data)





