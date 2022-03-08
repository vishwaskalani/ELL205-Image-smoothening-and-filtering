import cv2
from skimage.restoration import estimate_sigma

def estimate_noise(image_path):
    img = cv2.imread(image_path)
    return estimate_sigma(img, channel_axis=-1, average_sigmas=True)
print(estimate_noise("1.jpg"))
print(estimate_noise())