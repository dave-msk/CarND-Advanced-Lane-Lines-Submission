import numpy as np
import cv2


def abs_sobel_threshold(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
	"""
	Performs x or y directional derivative magnitude thresholding.
	
	Arguments
	---------
	       image: input image
	      orient: orientation, only 'x' and 'y' are allowed
	sobel_kernel: side length of Sobel kernel, must be odd number
	      thresh: threshold -- [lower, upper] (inclusive)
	"""
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	if orient=='x':
		sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	elif orient=='y':
		sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	else:
		raise ValueError("Only 'x' or 'y' are valid for named parameter 'orient'")
	
	abs_sobel = np.absolute(sobel)
	scaled = np.uint8(255*abs_sobel/np.max(abs_sobel))
	binary_output = np.zeros_like(scaled)
	binary_output[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
	return binary_output


def mag_threshold(image, sobel_kernel=3, mag_thresh=(0, 255)):
	"""
	Performs gradient magnitude thresholding.
	
	Arguments
	---------
	       image: input image
	sobel_kernel: side length of Sobel kernel, must be odd number
	  mag_thresh: magnitude threshold -- [lower, upper] (inclusive)
	"""
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	sobel = np.sqrt(sobelx**2 + sobely**2)
	scaled = np.uint8(255*sobel/np.max(sobel))
	binary_output = np.zeros_like(scaled)
	binary_output[(scaled >= mag_thresh[0]) & (scaled <= mag_thresh[1])] = 1
	return binary_output


def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
	"""
	Performs direction thresholding.
	
	Arguments
	---------
	       image: input image
	sobel_kernel: side length of Sobel kernel, must be odd number
	      thresh: angle threshold -- (lower, upper] (half-inclusive)
	"""
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	abs_sx = np.absolute(sobelx)
	abs_sy = np.absolute(sobely)
	orient = np.arctan2(sobely, sobelx)
	binary_output = np.zeros_like(orient)
	binary_output[(orient > thresh[0]) & (orient <= thresh[1])] = 1
	return binary_output


def gray_threshold(gray, thresh=(0, 255)):
	"""
	Performs thresholding on image with exactly one channel.
	
	Arguments
	---------
	    image: grayscale image
	   thresh: threshold -- [lower, upper] (inclusive)
	"""
	binary_output = np.zeros_like(gray)
	binary_output[(gray >= thresh[0]) & (gray <= thresh[1])] = 1
	return binary_output


def S_threshold(image, thresh=(0, 255)):
	"""
	Performs thresholding on S channel.

	Arguments
	---------
	    image: RGB image
	   thresh: saturation threshold -- [lower, upper] (inclusive)
	"""
	hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
	s_channel = hls[:,:,2]
	return gray_threshold(s_channel, thresh)


def R_threshold(image, thresh=(0, 255)):
	"""
	Performs thresholding on R channel.
	
	Arguments
	---------
	    image: RGB image
	   thresh: red channel threshold -- [lower, upper] (inclusive)
	"""
	r_channel = image[:,:,0]
	return gray_threshold(r_channel, thresh)


def H_threshold(image, thresh=(0,180)):
	"""
	Performs thresholding on hue channel.
	
	Arguments
	---------
	    image: RGB image
	   thresh: hue value range
	"""
	hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
	h_channel = hls[:,:,0]
	return gray_threshold(h_channel, thresh)


def SV_threshold(image, thresh_s=30, thresh_v=128):
	"""
	Performs thresholding on S and V channel.
	
	Arguments
	---------
	    image: RGB image 
	 thresh_s: upper threshold of S channel
	 thresh_v: lower threshold of V channel
	"""
	hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	binary = np.zeros(image.shape[:2])
	s_channel = hsv[:,:,1]
	v_channel = hsv[:,:,2]
	binary[(s_channel <= thresh_s) & (v_channel >= thresh_v)] = 1
	return binary

def thresholding_pipeline(undist, methods, args):
	"""
	Performs thresholding on undistorted image by every given 
	thresholding method and combine them.
	
	Arguments
	---------
	   undist: undistorted RGB image
	  methods: thresholding methods
	     args: arguments for thresholding methods
	"""
	binary_output = np.zeros(undist.shape[:2])
	for i in range(len(methods)):
		filtered = methods[i](undist, *args[i])
		binary_output[filtered > 0.5] = 1
	return binary_output
