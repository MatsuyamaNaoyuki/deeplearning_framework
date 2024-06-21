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




x = Variable(np.array([[1,2,3],[4,5,6]])) 
y = x.T
y.backward(retain_grad=True)
print(x.grad)

# c = Variable(np.array([10,20,30],[40,50,60]))
# t = x + c
# y = F.sum(x)
# print(y)