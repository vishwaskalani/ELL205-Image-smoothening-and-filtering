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
def plotlog(imagefile):
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
	typ="logsize"
	kern="LoG"
	img1=RGBify(cv2.imread(imagefile))
	logblur1=RGBify(cv2.imread(imagename+"_"+typ+"1.jpg"))
	logblur2=RGBify(cv2.imread(imagename+"_"+typ+"2.jpg"))
	logblur3=RGBify(cv2.imread(imagename+"_"+typ+"3.jpg"))
	logblur4=RGBify(cv2.imread(imagename+"_"+typ+"4.jpg"))
	logblur5=RGBify(cv2.imread(imagename+"_"+typ+"5.jpg"))
	plt.subplot(161),plt.imshow(img1.astype('uint8')),plt.title('Original')
	plt.xticks([]), plt.yticks([])
	plt.subplot(162),plt.imshow(logblur1.astype('uint8')),plt.title(kern+" ,3,1")
	plt.xticks([]), plt.yticks([])
	plt.subplot(163),plt.imshow(logblur2.astype('uint8')),plt.title(kern+" ,5,1")
	plt.xticks([]), plt.yticks([])
	plt.subplot(164),plt.imshow(logblur3.astype('uint8')),plt.title(kern+" ,7,1")
	plt.xticks([]), plt.yticks([])
	plt.subplot(165),plt.imshow(logblur4.astype('uint8')),plt.title(kern+" ,9,1")
	plt.xticks([]), plt.yticks([])
	plt.subplot(166),plt.imshow(logblur5.astype('uint8')),plt.title(kern+" ,11,1")
	plt.xticks([]), plt.yticks([])
	plt.show()
plotlog("1.jpg")