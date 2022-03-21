import numpy
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def kernel_plotter(Z):
    x = range(Z.shape[0])
    y = range(Z.shape[1])
    data = Z
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = numpy.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_wireframe(X, Y, data)
    plt.show()

def func1(x,mean,sigma):
    return math.exp(-((((x-mean)/(sigma))**2)/2))
function1 = numpy.vectorize(func1)
def func2(x,mean,sigma):
    return (sigma**2) - ((x-mean)**2)
function2 = numpy.vectorize(func2)

# def evaluator(x,mean,sigma):
#     a = math.exp(-((((x-mean)/(sigma))**2)/2))   
#     b = 
#     return (1 / (numpy.sqrt(2 * numpy.pi) * (sigma**3)))*a


# def gaussian_kernel_generator(n,sigma): 
#     kernel_1D = numpy.linspace(-1*(n // 2), n // 2, n)
#     for i in range(n):
#         kernel_1D[i] = evaluator(kernel_1D[i], 0, sigma)
#     ## To generate a 2d gaussian kernel, we can take an outer product of the 1d kernel with itself
#     kernel_2D = numpy.outer(kernel_1D.T, kernel_1D.T) 
#     kernel_2D *= 1.0 / kernel_2D.sum() 
#     return kernel_2D

# filter = gaussian_kernel_generator(5,1)
# print(filter)

def log_kernel_generator(n,sigma): 
    kernel_1D_x = numpy.linspace(-1*(n // 2), n // 2, n)
    kernel_1D_y = numpy.linspace(-1*(n // 2), n // 2, n)
    X,Y = numpy.meshgrid(kernel_1D_x, kernel_1D_y)
    c = (1 / ((2 * numpy.pi) * (sigma**6)))
    kernel_2D = -1*c*function1(X,0,sigma)*function1(Y,0,sigma)*(function2(X,0,sigma)+function2(Y,0,sigma))
    return kernel_2D

# x = numpy.linspace(-5, 5, 11)

# y = numpy.linspace(-5, 5, 11)

# # full coorindate arrays

# a,b = numpy.meshgrid(x, y)
# print(b)
a = log_kernel_generator(5,1)
print(a)