from matplotlib import pyplot as plt
import cv2
import numpy as np

def RGBify(tensor):
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
	return cv2.merge((b,g,r))
    
img1=RGBify(cv2.imread("1.jpg"))
blur=RGBify(cv2.imread("output.jpg"))
gblur=RGBify(cv2.imread("outputg.jpg"))

plt.subplot(131),plt.imshow(img1.astype('uint8')),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(blur.astype('uint8')),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(gblur.astype('uint8')),plt.title('Gaussian Blurred')
plt.xticks([]), plt.yticks([])
plt.show()