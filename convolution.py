from random import gauss
import numpy as np
import math
import cv2



def convolve(matrix, kernel):
    mRows = matrix.shape[0]
    mCols = matrix.shape[1]
    kRows = kernel.shape[0]
    kCols = kernel.shape[1]

    res = np.zeros(shape=(mRows - kRows+1, mCols - kCols+1))

    for i in range(mRows - kRows+1):
        for j in range(mCols - kCols+1):
            for a in range(kRows):
                for b in range(kCols):
                    res[i][j] += matrix[i+a][j+b] * kernel[a][b]

    return res


def convolveRGB(tensor, kernel):
    padding=kernel.shape[0]//2
    tensor= cv2.copyMakeBorder(tensor,padding,padding,padding,padding,cv2.BORDER_CONSTANT,value=(0,0,0))
    rows = tensor.shape[0]
    cols = tensor.shape[1]
    r = np.zeros(shape=(rows, cols))
    g = np.zeros(shape=(rows, cols))
    b = np.zeros(shape=(rows, cols))

    for i in range(rows):
        for j in range(cols):
            b[i][j] = tensor[i][j][0]
            g[i][j] = tensor[i][j][1]
            r[i][j] = tensor[i][j][2]

    rOut = convolve(r, kernel)
    gOut = convolve(g, kernel)
    bOut = convolve(b, kernel)
    result = cv2.merge((bOut,gOut,rOut))
    return result


#LOG------------------------------------------------------------------------


def  log_kernel(n,sigma):
    return gaussian_kernel(n,sigma)


#LOG --------------------------------------------------------------------


def evaluator(x,mean,sigma):
    a = math.exp(-((((x-mean)/(sigma))**2)/2))    
    return (1 / (np.sqrt(2 * np.pi) * sigma))*a


def gaussian_kernel(n,sigma): 
    kernel_1D = np.linspace(-1*(n // 2), n // 2, n)
    for i in range(n):
        kernel_1D[i] = evaluator(kernel_1D[i], 0, sigma)
    ## To generate a 2d gaussian kernel, we can take an outer product of the 1d kernel with itself
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T) 
    kernel_2D *= 1.0 / kernel_2D.sum() 
    return kernel_2D

def delta_kernel(n):
    filter = np.zeros(shape=(n, n))
    filter[1][1]=1
    return(filter)
def mean_kernel(n):
    filter = np.ones(shape=(n, n))/(n*n)
    return(filter)

def filter(image,filtertype,n,*args):
    if(len(args)>0):
        kernel={"Gaussian":gaussian_kernel(n,args[0]),"LoG":log_kernel(n,args[0])}
    else:
        kernel={"Delta":delta_kernel(n),"Mean":mean_kernel(n)}
    result = convolveRGB(image, kernel[filtertype])
    return(result)
