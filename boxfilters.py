import cv2
from skimage.restoration import estimate_sigma
import convolution
def filters(imagefile):
	img1 = cv2.imread(imagefile)
	imagename=imagefile.split(".")[0]
	dblur=convolution.filter(img1,"Delta",3)
	cv2.imwrite(imagename+"_delta.jpg",dblur)
	gblur =convolution.filter(img1,"Gaussian",3,456)
	cv2.imwrite(imagename+"_gaussian.jpg",gblur)
	logblur =convolution.filter(img1,"LoG",3,456)
	cv2.imwrite(imagename+"_log.jpg",logblur)
	mblur=convolution.filter(img1,"Mean",3)
	cv2.imwrite(imagename+"_mean.jpg",mblur)

	print(estimate_sigma(img1, channel_axis=-1, average_sigmas=True))
	print(estimate_sigma(dblur, channel_axis=-1, average_sigmas=True))
	print(estimate_sigma(gblur, channel_axis=-1, average_sigmas=True))
	print(estimate_sigma(mblur, channel_axis=-1, average_sigmas=True))
	print(estimate_sigma(logblur, channel_axis=-1, average_sigmas=True))
filters("2.jpg")
