if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from mydezero import Variable, Layer
from mydezero import Parameter
from mydezero import Function
from mydezero.utils import plot_dot_graph
import mydezero.layers as L
from mydezero.utils import sum_to
import mydezero.functions as F
import math
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

model = Layer()
model.l1 = L.Linear(5)
model.l2 = L.Linear(3)

for p in model.params():
    print(p)

model.cleargrads()


# class TwoLayerNet(Model):
#     def __init__(self, hidden_size, out_size):
#         super().__init__()
#         self.l1 = L.Linear(hidden_size)
#         self.l2 = L.Linear(out_size)

#     def forward(self, x):
#         y = F.sigmoid(self.l1(x))
#         y = self.l2(y)
#         return y
    
# x = Variable(np.random.randn(5,10), name='x')
# model = TwoLayerNet(100, 10)
# model.plot(x)
