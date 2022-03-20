from random import gauss
import numpy as np
import math
import cv2
def evaluator(x,mean,sigma):
    a = math.exp(-((((x-mean)/(sigma))**2)/2))    
    return (1 / (np.sqrt(2 * np.pi) * sigma))*a


def gaussian_kernel_generator(n,sigma): 
    kernel_1D = np.linspace(-1*(n // 2), n // 2, n)
    for i in range(n):
        kernel_1D[i] = evaluator(kernel_1D[i], 0, sigma)
    ## To generate a 2d gaussian kernel, we can take an outer product of the 1d kernel with itself
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T) 
    kernel_2D *= 1.0 / kernel_2D.sum() 
    return kernel_2D


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


def gaussianBlur(image):
    filter = gaussian_kernel_generator(3,456)
    result = convolveRGB(image, filter)
    cv2.imwrite("outputg.jpg", result)
    return(result)

def linearBlur(image):
    filter = np.zeros(shape=(3, 3))
    filter[1][1]=1
    result = convolveRGB(image, filter)
    cv2.imwrite("output.jpg", result)
    return(result)
