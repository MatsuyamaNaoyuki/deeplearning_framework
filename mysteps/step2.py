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





np.random.seed(0)
x = np.random.rand(100,1)
y = np.sin(2 * np.pi * x) + np.random.rand(100,1)
 
l1 = L.Linear(10)
l2 = L.Linear(1)

def predict(x):
    y = l1(x)
    y = F.sigmoid(y)
    y = l2(y)
    return y

lr = 0.2
iters = 10000
for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    l1.cleargrads()
    l2.cleargrads()
    loss.backward()

    for l in [l1, l2]:
        for p in l.params():
            p.data -= lr * p.grad.data
    if i % 1000 == 0:
        print(loss)



print(x.size)

print(y.size)
newy = predict(x)
 

print(newy.data.size)
plt.scatter(x, y)
plt.scatter(x, newy.data)

plt.show()
