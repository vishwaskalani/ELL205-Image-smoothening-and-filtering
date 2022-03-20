# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
# from PIL import Image, ImageFilter
# #%matplotlib inline
# image = cv2.imread('AM04NES.JPG') # reads the image
# image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # convert to HSV
# print(image.dtype )
# figure_size = 9 # the dimension of the x and y axis of the kernal.
# new_image = cv2.blur(image,(figure_size, figure_size))
# plt.figure(figsize=(11,6))
# plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_HSV2RGB)),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)),plt.title('Mean filter')
# plt.xticks([]), plt.yticks([])
# plt.show()



from operator import index
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.restoration import estimate_sigma
import convolution

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

def estimate_noise(image_path):
    img = cv2.imread(image_path)
    return estimate_sigma(img, channel_axis=-1, average_sigmas=True)

img1 = cv2.imread("1.jpg")
blur=convolution.linearBlur(img1)
gblur =convolution.gaussianBlur(img1)
print(img1.shape[0],img1.shape[1],img1.shape[2])
print(blur.shape[0],blur.shape[1],blur.shape[2])
print(gblur.shape[0],gblur.shape[1],gblur.shape[2])
print(estimate_sigma(img1, channel_axis=-1, average_sigmas=True))
print(estimate_sigma(blur, channel_axis=-1, average_sigmas=True))
print(estimate_sigma(gblur, channel_axis=-1, average_sigmas=True))
img1=RGBify(img1)
blur=RGBify(blur)
gblur=RGBify(gblur)
plt.subplot(131),plt.imshow(img1.astype('uint8')),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(gblur.astype('uint8')),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(blur.astype('uint8')),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()


