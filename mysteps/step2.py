if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from mydezero import Variable
from mydezero.utils import plot_dot_graph
def sphere(x,y):
    z = x ** 2 + y ** 2
    return z

def mytans(x,y):
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return z

x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
y = x0 + x1

z = mytans (x0, x1)
z.backward()
x0.name = x0
x1.name = x1
z.name = z

plot_dot_graph(z, verbose=False, to_file='mytans.png')
