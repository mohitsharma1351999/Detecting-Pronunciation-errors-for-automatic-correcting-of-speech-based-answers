import numpy as np
import matplotlib.pyplot as plt
import math

n = 1000
k = [1, 2, 4, 8]
a = [0.4, 2.5, 5]


def sinplot(n, k, a):
    t = np.arange(n)
    y = np.sin((2 * np.pi * k * t)/n)
    legend_handle, = plt.plot(t, y)
    plt.legend([legend_handle], ["k = " + str(k) + ', a = ' + str(a)])
    plt.xlabel('time')
    plt.ylabel('y')
    plt.show()


for i in k:
    for j in a:
        sinplot(n, i, j)
