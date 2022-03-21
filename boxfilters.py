import cv2
from skimage.restoration import estimate_sigma
import convolution
def filters(imagefile):
	img1 = cv2.imread(imagefile)
	dblur=convolution.filter(img1,"Delta",3)
	gblur =convolution.filter(img1,"Gaussian",3,456)
	mblur=convolution.filter(img1,"Mean",3)
	cv2.imwrite(imagefile+"_delta.jpg",dblur)
	cv2.imwrite(imagefile+"_gaussian.jpg",gblur)
	cv2.imwrite(imagefile+"_mean.jpg",mblur)
	print(gblur.shape[0],gblur.shape[1])

	print(estimate_sigma(img1, channel_axis=-1, average_sigmas=True))
	print(estimate_sigma(dblur, channel_axis=-1, average_sigmas=True))
	print(estimate_sigma(gblur, channel_axis=-1, average_sigmas=True))
	print(estimate_sigma(mblur, channel_axis=-1, average_sigmas=True))
filters("1.jpg")




