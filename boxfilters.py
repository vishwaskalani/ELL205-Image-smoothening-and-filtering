import cv2
from skimage.restoration import estimate_sigma
import convolution
def filters(imagefile):
	img1 = cv2.imread(imagefile)
	imagename=imagefile.split(".")[0]
	logblur1=convolution.filter(img1,"LoG",5,0.6)
	cv2.imwrite(imagename+"_log1.jpg",logblur1)
	logblur2 =convolution.filter(img1,"LoG",13,0.67)
	cv2.imwrite(imagename+"_log2.jpg",logblur2)
	logblur3 =convolution.filter(img1,"LoG",3,0.5)
	cv2.imwrite(imagename+"_log3.jpg",logblur3)
	logblur4=convolution.filter(img1,"LoG",5,0.6)
	cv2.imwrite(imagename+"_log4.jpg",logblur4)
	logblur5 =convolution.filter(img1,"LoG",5,0.55)
	cv2.imwrite(imagename+"_log5.jpg",logblur5)
	logblur6 =convolution.filter(img1,"LoG",5,0.5)
	cv2.imwrite(imagename+"_log6.jpg",logblur6)


	print(estimate_sigma(img1, channel_axis=-1, average_sigmas=True))
	print(estimate_sigma(logblur1, channel_axis=-1, average_sigmas=True))
	print(estimate_sigma(logblur2, channel_axis=-1, average_sigmas=True))
	print(estimate_sigma(logblur3, channel_axis=-1, average_sigmas=True))
	print(estimate_sigma(logblur4, channel_axis=-1, average_sigmas=True))
	print(estimate_sigma(logblur5, channel_axis=-1, average_sigmas=True))
	print(estimate_sigma(logblur6, channel_axis=-1, average_sigmas=True))
filters("5.jpg")
