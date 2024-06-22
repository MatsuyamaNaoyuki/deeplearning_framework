if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from mydezero import Variable
from mydezero import Parameter
from mydezero import Function
from mydezero.utils import plot_dot_graph
import mydezero.layers as L
from mydezero.utils import sum_to
import mydezero.functions as F
import math
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

model = L.Layer()
model.l1 = L.Linear(5)
model.l2 = L.Linear(3)

def predict(model, x):
    y = model.l1(x)
    y = F.sigmoid(y)
    y = model.l2(y)
    return y

for p in model.params():
    print(p)

model.cleargrads()