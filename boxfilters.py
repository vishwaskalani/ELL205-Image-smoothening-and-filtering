import cv2
from skimage.restoration import estimate_sigma
import convolution
def filters(imagefile):
	img1 = cv2.imread(imagefile)
	imagename=imagefile.split(".")[0]
	typ="logsize"
	logblur1=convolution.filter(img1,"LoG",3,1)
	cv2.imwrite(imagename+"_"+typ+"1.jpg",logblur1)
	logblur2 =convolution.filter(img1,"LoG",5,1)
	cv2.imwrite(imagename+"_"+typ+"2.jpg",logblur2)
	logblur3 =convolution.filter(img1,"LoG",7,1)
	cv2.imwrite(imagename+"_"+typ+"3.jpg",logblur3)
	logblur4=convolution.filter(img1,"LoG",9,1)
	cv2.imwrite(imagename+"_"+typ+"4.jpg",logblur4)
	logblur5 =convolution.filter(img1,"LoG",11,1)
	cv2.imwrite(imagename+"_"+typ+"5.jpg",logblur5)

	print(estimate_sigma(img1, channel_axis=-1, average_sigmas=True))
	print(estimate_sigma(logblur1, channel_axis=-1, average_sigmas=True))
	print(estimate_sigma(logblur2, channel_axis=-1, average_sigmas=True))
	print(estimate_sigma(logblur3, channel_axis=-1, average_sigmas=True))
	print(estimate_sigma(logblur4, channel_axis=-1, average_sigmas=True))
	print(estimate_sigma(logblur5, channel_axis=-1, average_sigmas=True))
filters("1.jpg")
