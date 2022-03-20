import numpy
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# # Set up grid and test data
# nx, ny = 4,4
# x = range(nx)
# y = range(ny)

# data = numpy.random.random((nx, ny))

# hf = plt.figure()
# ha = hf.add_subplot(111, projection='3d')

# X, Y = numpy.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
# ha.plot_surface(X, Y, data)

# plt.show()

def kernel_plotter(Z):
    x = range(Z.shape[0])
    y = range(Z.shape[1])
    data = Z
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = numpy.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, data)
    plt.show()

def gen_gaussian_kernel(shape, mean, var):
    coors = [range(shape[d]) for d in range(len(shape))]
    k = numpy.zeros(shape=shape)
    cartesian_product = [[]]
    for coor in coors:
        cartesian_product = [x + [y] for x in cartesian_product for y in coor]
    for c in cartesian_product:
        s = 0
        for cc, m in zip(c,mean):
            s += (cc - m)**2
        k[tuple(c)] = numpy.exp(-s/(2*var))
    l = numpy.sum(k)
    k = k/l
    return k
# filter = gen_gaussian_kernel(shape=(5,5),mean=(1,1),var=1.0)
# print(filter)

def evaluator(x,mean,sigma):
    a = math.exp(-((((x-mean)/(sigma))**2)/2))    
    return (1 / (numpy.sqrt(2 * numpy.pi) * sigma))*a


def gaussian_kernel_generator(n,sigma): 
    kernel_1D = numpy.linspace(-1*(n // 2), n // 2, n)
    for i in range(n):
        kernel_1D[i] = evaluator(kernel_1D[i], 0, sigma)
    ## To generate a 2d gaussian kernel, we can take an outer product of the 1d kernel with itself
    kernel_2D = numpy.outer(kernel_1D.T, kernel_1D.T) 
    kernel_2D *= 1.0 / kernel_2D.sum() 
    return kernel_2D

filter = gaussian_kernel_generator(5,1)
print(filter)