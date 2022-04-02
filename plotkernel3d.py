import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def kernel_plotter(Z):
    x = range(Z.shape[0])
    y = range(Z.shape[1])
    data = Z
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_wireframe(X, Y, data,rstride=1, cstride=1, color='maroon')
    plt.show()

