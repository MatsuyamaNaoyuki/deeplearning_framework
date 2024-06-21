if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from mydezero import Variable
from mydezero import Function
from mydezero.utils import plot_dot_graph

from mydezero.utils import sum_to
import mydezero.functions as F
import math
import matplotlib.pyplot as plt


def sphere(x,y):
    z = x ** 2 + y ** 2
    return z

def mytans(x,y):
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return z




x0 = Variable(np.array([[1,2,3]]))
x1 = Variable(np.array([10]))
y = x0 - x1
print(y)

y.backward()
print(x1.grad)

