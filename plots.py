from matplotlib import pyplot as plt
import cv2
import numpy as np
def plot(imagefile):
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
	imagename=imagefile.split('.')[0]
	img1=RGBify(cv2.imread(imagefile))
	dblur=RGBify(cv2.imread(imagename+"_delta.jpg"))
	gblur=RGBify(cv2.imread(imagename+"_gaussian.jpg"))
	logblur=RGBify(cv2.imread(imagename+"_log.jpg"))
	mblur=RGBify(cv2.imread(imagename+"_mean.jpg"))

	plt.subplot(151),plt.imshow(img1.astype('uint8')),plt.title('Original')
	plt.xticks([]), plt.yticks([])
	plt.subplot(152),plt.imshow(dblur.astype('uint8')),plt.title('Original 2.0')
	plt.xticks([]), plt.yticks([])
	plt.subplot(153),plt.imshow(gblur.astype('uint8')),plt.title('Gaussian Blurred')
	plt.xticks([]), plt.yticks([])
	plt.subplot(154),plt.imshow(logblur.astype('uint8')),plt.title('LoG Blurred')
	plt.xticks([]), plt.yticks([])
	plt.subplot(155),plt.imshow(mblur.astype('uint8')),plt.title('Mean Blurred')
	plt.xticks([]), plt.yticks([])
	plt.show()
plot("1.jpg")