if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from mydezero import Variable
from mydezero import Function
from mydezero.utils import plot_dot_graph
import mydezero.functions as F
import math
import matplotlib.pyplot as plt


def sphere(x,y):
    z = x ** 2 + y ** 2
    return z

def mytans(x,y):
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return z



x = Variable(np.array(2.0))
y = x**2
y.backward(create_graph=True)
gx = x.grad
x.cleargrad()

z = gx ** 3 + y
z.backward()
print(x.grad)
