import numpy as np
import cv2
def gen_gaussian_kernel(shape, mean, var):
    coors = [range(shape[d]) for d in range(len(shape))]
    k = np.zeros(shape=shape)
    cartesian_product = [[]]
    for coor in coors:
        cartesian_product = [x + [y] for x in cartesian_product for y in coor]
    for c in cartesian_product:
        s = 0
        for cc, m in zip(c,mean):
            s += (cc - m)**2
        k[tuple(c)] = np.exp(-s/(2*var))
    l = np.sum(k)
    k = k/l
    return k


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
    tensor= cv2.copyMakeBorder(tensor,1,1,1,1,cv2.BORDER_CONSTANT,value=(0,0,0))
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
    filter = gen_gaussian_kernel(shape=(3,3),mean=(1,1),var=1.0)
    result = convolveRGB(image, filter)
    cv2.imwrite("outputg.jpg", result)
    return(result)

def linearBlur(image):
    filter = np.zeros(shape=(3, 3))
    filter[1][1]=1
    result = convolveRGB(image, filter)
    cv2.imwrite("output.jpg", result)
    return(result)
