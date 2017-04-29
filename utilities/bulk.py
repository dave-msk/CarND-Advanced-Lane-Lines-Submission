import cv2


def load_images(image_names):
	"""
	Loads multiple images.
	
	Arguments
	---------
	    image_names: list of filename of images

	Returns: list of loaded images
	"""
	images = []
	for name in image_names:
		images.append(cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB))
	return images


def undistort_images(images, cam_mtx, dist):
	"""
	Performs undistortion to given images.
	
	Arguments
	---------
	    images: list of distorted images to be processed
	   cam_mtx: (intrinsic) camera matrix
	      dist: (intrinsic) distortion coefficients
	
	Returns: list of undistorted images
	"""
	undists = []
	for img in images:
		undists.append(cv2.undistort(img, cam_mtx, dist, None, cam_mtx))
	return undists


def warp_images(images, map_mtx, image_size):
	"""
	Performs perspective transform to given images.
	
	Arguments
	---------
	    images: list of undistorted images to be processed
	   map_mtx: map matrix from cv2.getPerspectiveTransform()
	image_size: size of output image -- (height, width)
	
	Returns: list of perspective transformed images
	"""
	result = []
	for img in images:
		warped = cv2.warpPerspective(img, map_mtx, image_size[::-1])
		result.append(warped)
	return result


def apply_func(image, method, args):
	return method(image, *args)


def apply_all(images, method, args):
	results = []
	for image in images:
		result = apply_func(image, method, args)
		results.append(result)
	return results


def combine_extracts(binary_extracts):
	binary_images = []
	image_size = binary_extracts[0][0].shape
	for i in range(len(binary_extracts[0])):
		binary_image = np.zeros(image_size)
		for j in range(len(binary_extracts)):
			extract = binary_extracts[j][i]
			binary_image[extract > 0.5] = 1
		binary_images.append(binary_image)
	return binary_images


def imshow(images, config, titles=None, save_as=None, show=False):
	length = min(len(images), config[0]*config[1])
	plt.figure(figsize=(config[1]*18,10*config[0]))
	for i in range(length):
		image = images[i]
		ax = plt.subplot(config[0],config[1],i+1)
		if image.shape[-1] == 3:
			ax.imshow(image)
		else:
			ax.imshow(image, cmap='gray')
		ax.axis('off')
		if titles and len(titles) > i:
			ax.set_title(titles[i], fontsize=30)
	plt.subplots_adjust(wspace=0.1, hspace=0.1)
	if save_as: plt.savefig(save_as, bbox_inches='tight')
	if show: plt.show()


def plot_search_results(lane_data, config, titles=None, save_as=None, show=False):
	length = min(len(lane_data), config[0]*config[1])
	if length == 0: return
	plt.figure(figsize=(config[1]*18,10*config[0]))
	height = lane_data[0][2].shape[0]
	ploty = np.linspace(0, height-1, height)
	for i in range(length):
		leftx, lefty = lane_data[i][0]
		left_fit = np.polyfit(lefty, leftx, 2)
		rightx, righty = lane_data[i][1]
		right_fit = np.polyfit(righty, rightx, 2)
		vis_img = lane_data[i][2]
		left_fitx = left_fit[0]*(ploty**2) + left_fit[1]*ploty + left_fit[2]
		right_fitx = right_fit[0]*(ploty**2) + right_fit[1]*ploty + right_fit[2]
		ax = plt.subplot(config[0],config[1],i+1)
		ax.imshow(vis_img)
		if titles and len(titles) > i:
			ax.set_title(titles[i], fontsize=30)
		ax.plot(left_fitx, ploty, color='yellow')
		ax.plot(right_fitx, ploty, color='yellow')
	plt.subplots_adjust(wspace=0.1,hspace=0.1)
	if save_as: plt.savefig(save_as, bbox_inches='tight')
	if show: plt.show()
