import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set up grid and test data
nx, ny = 3, 3
x = range(nx)
y = range(ny)

data = numpy.random.random((nx, ny))

hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')

X, Y = numpy.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
ha.plot_surface(X, Y, data)

plt.show()
# from mpl_toolkits.mplot3d import axes3d
# import matplotlib.pyplot as plt
# import numpy as np

# def kernel_plotter(Z):
#     x = np.arange(0,Z.shape[0])
#     y = np.arange(0,Z.shape[1])
#     X,Y = np.meshgrid(x,y)
#     fig = plt.figure(figsize=(6,6))
#     ax = fig.add_subplot(111, projection='3d')
#     # Plot a 3D surface
#     ax.plot_surface(X, Y, Z)
#     plt.show()
# arr = np.array([[1,2,3],[4,5,6]])
# kernel_plotter(arr)

