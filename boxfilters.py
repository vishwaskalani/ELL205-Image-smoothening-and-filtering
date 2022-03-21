import cv2
from skimage.restoration import estimate_sigma
import convolution

img1 = cv2.imread("1.jpg")
blur=convolution.linearBlur(img1,3)
gblur =convolution.gaussianBlur(img1,3,456)

print(gblur.shape[0],gblur.shape[1])

print(estimate_sigma(img1, channel_axis=-1, average_sigmas=True))
print(estimate_sigma(blur, channel_axis=-1, average_sigmas=True))
print(estimate_sigma(gblur, channel_axis=-1, average_sigmas=True))




