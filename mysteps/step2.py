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
import matplotlib.pyplot as plt


np.random.seed(0)
x = np.random.rand(100,1)
y = 5 + 2 * x + np.random.rand(100,1)

x_np = x
y_np = y

x, y = Variable(x), Variable(y)

W = Variable(np.zeros((1,1)))
b = Variable(np.zeros(1))

def predict(x):
    y = F.matmul(x, W) + b
    return y

def mean_squared_error(x0, x1):
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)

lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    loss = mean_squared_error(y, y_pred)

    W.cleargrad()
    b.cleargrad()

    loss.backward()

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data
    print(W, b, loss)



xnew = np.arange(0, 1, 0.01)   # xを-5から5まで0.1刻みで用意する
ynew = b.data + W.data *xnew   
plt.scatter(x_np, y_np)
plt.scatter(xnew, ynew)
plt.show()
