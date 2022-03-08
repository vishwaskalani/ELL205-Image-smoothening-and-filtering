import numpy as np
import cv2


def readIm(filename):
    return cv2.imread(filename)


def convolve(matrix, kernel):
    mRows = matrix.shape[0]
    mCols = matrix.shape[1]
    kRows = kernel.shape[0]
    kCols = kernel.shape[1]

    res = np.zeros(shape=(mRows - kRows, mCols - kCols))

    for i in range(mRows - kRows):
        for j in range(mCols - kCols):
            for a in range(kRows):
                for b in range(kCols):
                    res[i][j] += matrix[i+a][j+b] * kernel[a][b]

    return res


def convolveRGB(tensor, kernel):
    rows = tensor.shape[0]
    cols = tensor.shape[1]
    r = np.zeros(shape=(rows, cols))
    g = np.zeros(shape=(rows, cols))
    b = np.zeros(shape=(rows, cols))

    for i in range(rows):
        for j in range(cols):
            r[i][j] = tensor[i][j][0]
            g[i][j] = tensor[i][j][1]
            b[i][j] = tensor[i][j][2]

    rOut = convolve(r, kernel)
    gOut = convolve(g, kernel)
    bOut = convolve(b, kernel)

    outRows = rOut.shape[0]
    outCols = rOut.shape[1]

    result = np.zeros(shape=(outRows, outCols, 3))
    for i in range(outRows):
        for j in range(outCols):
            result[i][j][0] = rOut[i][j]
            result[i][j][1] = gOut[i][j]
            result[i][j][2] = bOut[i][j]

    return result


def linearBlur(fileName):
    im = readIm(filename=fileName)
    filter = np.ones(shape=(3, 3))/9
    result = convolveRGB(im, filter)
    cv2.imwrite("output.jpg", result)


linearBlur("1.jpg")
