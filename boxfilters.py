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


import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.restoration import estimate_sigma

def estimate_noise(image_path):
    img = cv2.imread(image_path)
    return estimate_sigma(img, channel_axis=-1, average_sigmas=True)
img = cv2.imread('1.jpg')
print(type(img) )

blur = cv2.blur(img,(5,5))

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()
print(estimate_noise("1.jpg"))
print(estimate_noise(img))

def square_matrix(square):
	""" This function will calculate the value x
	(i.e. blurred pixel value) for each 3 * 3 blur image.
	"""
	tot_sum = 0
	
	# Calculate sum of all the pixels in 3 * 3 matrix
	for i in range(3):
		for j in range(3):
			tot_sum += square[i][j]
			
	return tot_sum // 9	 # return the average of the sum of pixels

def boxBlur(image):
	"""
	This function will calculate the blurred
	image for given n * n image.
	"""
	square = []	 # This will store the 3 * 3 matrix
				# which will be used to find its blurred pixel
				
	square_row = [] # This will store one row of a 3 * 3 matrix and
					# will be appended in square
					
	blur_row = [] # Here we will store the resulting blurred
					# pixels possible in one row
					# and will append this in the blur_img
	
	blur_img = [] # This is the resulting blurred image
	
	# number of rows in the given image
	n_rows = len(image)
	
	# number of columns in the given image
	n_col = len(image[0])
	
	# rp is row pointer and cp is column pointer
	rp, cp = 0, 0
	
	# This while loop will be used to
	# calculate all the blurred pixel in the first row
	while rp <= n_rows - 3:
		while cp <= n_col-3:
			
			for i in range(rp, rp + 3):
				
				for j in range(cp, cp + 3):
					
					# append all the pixels in a row of 3 * 3 matrix
					square_row.append(image[i][j])
					
				# append the row in the square i.e. 3 * 3 matrix
				square.append(square_row)
				square_row = []
			
			# calculate the blurred pixel for given 3 * 3 matrix
			# i.e. square and append it in blur_row
			blur_row.append(square_matrix(square))
			square = []
			
			# increase the column pointer
			cp = cp + 1
		
		# append the blur_row in blur_image
		blur_img.append(blur_row)
		blur_row = []
		rp = rp + 1 # increase row pointer
		cp = 0 # start column pointer from 0 again
	
	# Return the resulting pixel matrix
	return blur_img

# Driver code
image = [[7, 4, 0, 1],
		[5, 6, 2, 2],
		[6, 10, 7, 8],
		[1, 4, 2, 0]]
		
print(boxBlur(image))


