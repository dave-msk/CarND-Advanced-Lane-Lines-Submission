import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
import sys

from utilities.line import Line
from utilities.bulk import load_images
from utilities.thresholding import abs_sobel_threshold, S_threshold, R_threshold, H_threshold, SV_threshold, thresholding_pipeline
from utilities.sanity_check import *
from utilities.camera import *
from moviepy.editor import VideoFileClip

cal_dir = 'camera_cal/'
cal_names = 'calibration*.jpg'
test_dir = 'test_images/'
fig_dir = 'figures/'
output_dir = 'output_images/'
cal_data_name = 'cal.p'

rc_text = "Radius of Curvature = %d(m)"
ofs_text = "Vehicle is %.2fm %s of center"

image_size = (720, 1280)
pattern_size = (9, 6)

pt_src = np.float32(
          [[202, 720],
           [1103, 720],
           [701, 460],
           [580, 460]])

pt_dst = np.float32(
          [[250, 720],
           [1030, 720],
           [1030, 0],
           [250, 0]])

nwindows = 9
margin = 70
minpix = 200
road_length = 39
road_width = 3.7
n_cache = 5
max_failed_count = 3

left_lane = Line(n_cache)
right_lane = Line(n_cache)
mid_lane = Line(n_cache)

ym_per_pix = road_length/720
xm_per_pix = road_width/781
y_eval_pix = image_size[0]-1
y_eval = y_eval_pix*ym_per_pix
x_mid_pos_pix = image_size[1]
x_mid_pos = (x_mid_pos_pix//2)*xm_per_pix


def sliding_window_search(binary_warped, nwindows=9, margin=100, minpix=50, left=True):
	"""
	Performs sliding window to collect points close to left or right lane line.
	
	Arguments
	---------
	binary_warped: Warped binary image for lane lines extraction
	     nwindows: Number of sliding windows (vertically)
	       margin: Half width of the windows
	       minpix: Minimum number of pixels found to recenter window
	         left: True to find left lane line, False to find the right one
	
	Returns: lane_x, lane_y
	-------
	 lane_x: x coordinates of white pixels close to desired lane line
	 lane_y: y coordinates of white pixels close to desired lane line
	"""
	# Define convolution window
	conv_win = np.ones(margin)
	
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
	
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = histogram.shape[0]//2
	if left:
		lane_base = np.argmax(np.convolve(conv_win, histogram[:midpoint])) - margin//2
	else:
		lane_base = np.argmax(np.convolve(conv_win, histogram[midpoint:])) - margin//2 + midpoint
	
	# Set height of windows
	window_height = np.int(binary_warped.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	x_current = lane_base
	
	# Create empty lists to receive left and right lane pixel indices
	lane_inds = []
	
	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = binary_warped.shape[0] - (window+1)*window_height
		win_y_high = binary_warped.shape[0] - window*window_height
		win_x_low = x_current - margin
		win_x_high = x_current + margin
		good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

		# Append these indices to the lists
		lane_inds.append(good_inds)
		
		# If more than minpix pixels are found, recenter next window on their mean position
		if len(good_inds) > minpix:
			x_current = np.int(np.mean(nonzerox[good_inds]))

	# Concatenate the arrays of indices
	lane_inds = np.concatenate(lane_inds)
	
	# Extract left and right line pixel positions
	lane_x = nonzerox[lane_inds] if lane_inds.any() else None
	lane_y = nonzeroy[lane_inds] if lane_inds.any() else None
	
	return lane_x, lane_y

def fine_tune_search(binary_warped, lane_fit, margin=100):
	"""
	Performs 2nd order polynomial fitting to describe left and right
	lane lines by fine-tuning corresponding polynomials of the previous
	frame.
	
	Arguments
	---------
	binary_warped: Warped binary image for lane lines extraction
	     lane_fit: Coefficients of the polynomial of the lane
	               line in the previous frame
	       margin: Half width of the windows
	
	Returns: lane_x, lane_y
	-------
	 lane_x: x coordinates of white pixels close to the detected lane line
	 lane_y: y coordinates of white pixels close to the detected lane line
	"""
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	lane_centerx = lane_fit[0]*(nonzeroy**2) + lane_fit[1]*nonzeroy + lane_fit[2]
	lane_inds = ((nonzerox > lane_centerx - margin) & (nonzerox < lane_centerx + margin))
	
	# Extract lane line pixel positions
	lane_x = nonzerox[lane_inds] if lane_inds.any() else None
	lane_y = nonzeroy[lane_inds] if lane_inds.any() else None
	
	return lane_x, lane_y


def polyfit(left_lane_pix, right_lane_pix):
	"""
	Fits polynomials for the left and right lane lines.
	
	Arguments
	---------
	    left_lane_pix: Coordinates of left lane pixels
	   right_lane_pix: Coordinates of right lane pixels
	
	Returns: left_fit, right_fit, mid_fit
	-------
	    left_fit: Coefficients of polynomial fitted to the left lane
	   right_fit: Coefficients of polynomial fitted to the right lane
	     mid_fit: Coefficients of polynomial fitted to the lane center
	"""
	left_fit = np.polyfit(left_lane_pix[1], left_lane_pix[0], 2)
	right_fit = np.polyfit(right_lane_pix[1], right_lane_pix[0], 2)
	mid_fit = np.mean(np.array([left_fit, right_fit]), axis=0)
	return left_fit, right_fit, mid_fit


def radius_curvature(poly_fit, y_eval):
	"""
	Calculates the radius of curvature
	
	Arguments
	---------
	    poly_fit: Coefficients of a quadratic function
	      y_eval: Evaluation point
	
	Returns: Radius of curvature of the quadratic curve at y_eval
	"""
	return ((1 + (2*poly_fit[0]*y_eval + poly_fit[1])**2)**1.5)/np.absolute(2*poly_fit[0])


def offset(lane_fit, y_eval, x_pos):
	"""
	Calculates the offset of fitted curve from given x-position
	
	Arguments
	---------
	    lane_fit: Coefficients of the polynomial fitted to the lane center
	      y_eval: Evalution point
	       x_pos: x position of vehicle
	"""
	x_lane_pos = lane_fit[0]*y_eval**2 + lane_fit[1]*y_eval + lane_fit[2]
	return x_pos - x_lane_pos


def project_lane(binary_warped, image, inv_mtx, left_fit, right_fit, margin):
	"""
	Projects the detected lane to undistorted original image
	
	Arguments
	---------
	binary_warped: Warped binary image from thresholding pipeline
	        image: Undistorted original image
	      inv_mtx: Matrix for inverse perspective transform
	     left_fit: Polynomial coefficients of left lane line
	    right_fit: Polynomial coefficients of right lane line
	       margin: Margin for coloring lane line pixels detected in binary_warped
	
	Returns: Undistorted original image with lane line pixels colored
	         and lane projected.
	"""
	# Create an image to draw the lines on
	color_left = np.zeros_like(image).astype(np.uint8)
	color_right = np.copy(color_left)
	project_warp = np.copy(color_left)
	#color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
	
	# Calculate lane line points
	ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	
	# Paint left and right lane lines
	# Recast the x and y points into usable format for cv2.fillPoly()
	# Then draw the lane onto the warped image
	
	# Left lane line:
	pts_left_lane_l = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	pts_left_lane_r = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
	pts_left_lane = np.hstack((pts_left_lane_l, pts_left_lane_r))
	cv2.fillPoly(color_left, np.int_([pts_left_lane]), (255,0,0))
	
	# Right lane line:
	pts_right_lane_l = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	pts_right_lane_r = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
	pts_right_lane = np.hstack((pts_right_lane_l, pts_right_lane_r))
	cv2.fillPoly(color_right, np.int_([pts_right_lane]), (0,0,255))
	
	# Lane projection:
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))
	
	cv2.fillPoly(project_warp, np.int_([pts]), (0,255,0))
	
	# Mask out regions without actual lane lines
	color_left[binary_warped < 0.5] = [0,0,0]
	color_right[binary_warped < 0.5] = [0,0,0]
	color_warp = color_left + color_right + project_warp
	
	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, inv_mtx, (image.shape[1], image.shape[0]))
	# Combine the result with the original image
	result = cv2.addWeighted(image, 0.7, newwarp, 0.3, 0)
	return result


def side_length(lane_fit, y_eval):
	"""
	Calculates the length of chord.
	"""
	return np.sqrt((lane_fit[0]*y_eval**2 + lane_fit[1]*y_eval)**2 + y_eval**2)


def process_image(image):
	"""
	Process image through the lane line detection pipeline.
	
	Arguments
	---------
	    image: Front image from a vehicle
	"""
	# Undistort image
	undist = cv2.undistort(image, mtx, dist, None, mtx)
	
	# Extract binary image
	binary = thresholding_pipeline(undist, threshold_methods, threshold_args)
	
	# Perform perspective transform to obtain bird's eye view
	binary_warped = cv2.warpPerspective(binary, map_mtx, image_size[::-1])
	
	# Search lane line points
	left_reset_flag = left_lane.need_reset(max_failed_count)
	right_reset_flag = right_lane.need_reset(max_failed_count)
	
	if left_reset_flag:
		left_lane_pix = sliding_window_search(binary_warped, margin=margin, minpix=minpix, left=True)
	else:
		left_lane_pix = fine_tune_search(binary_warped, lane_fit=left_lane.get_last_fit_pix(), margin=margin)
	
	if right_reset_flag:
		right_lane_pix = sliding_window_search(binary_warped, margin=margin, minpix=minpix, left=False)
	else:
		right_lane_pix = fine_tune_search(binary_warped, lane_fit=right_lane.get_last_fit_pix(), margin=margin)
		
	if left_lane_pix[0] == None or right_lane_pix[0] == None:
		left_lane.set_detected(False)
		right_lane.set_detected(False)
		mid_lane.set_detected(False)
		smoothed_mid_rc = mid_lane.get_smoothed_roc()
		rc_message = rc_text % (int(smoothed_mid_rc))
		lane_projected = cv2.addWeighted(undist, 0.7, np.zeros_like(undist), 0, 0)
		cv2.putText(lane_projected, rc_message, (50, 49), cv2.	FONT_HERSHEY_SIMPLEX, 2, (255,)*3, 2)
		return lane_projected
	
	# Fit lane line polynomials in pixel scale
	left_fit_pix, right_fit_pix, mid_fit_pix = polyfit(left_lane_pix, right_lane_pix)
	
	# Fit lane line polynomials in real world scale
	left_lane_pix_scaled = (left_lane_pix[0]*xm_per_pix, left_lane_pix[1]*ym_per_pix)
	right_lane_pix_scaled = (right_lane_pix[0]*xm_per_pix, right_lane_pix[1]*ym_per_pix)
	left_fit, right_fit, mid_fit = polyfit(left_lane_pix_scaled, right_lane_pix_scaled)

	left_rc = radius_curvature(left_fit, y_eval)
	right_rc = radius_curvature(right_fit, y_eval)
	mid_rc = radius_curvature(mid_fit, y_eval)

	left_offset = offset(left_fit, y_eval, x_mid_pos)
	right_offset = offset(right_fit, y_eval, x_mid_pos)
	mid_offset = offset(mid_fit, y_eval, x_mid_pos)
	
	# Perform Sanity check
	left_d = side_length(left_fit, y_eval)
	right_d = side_length(right_fit, y_eval)
	
	sanity_args = [(left_fit_pix, right_fit_pix, y_eval_pix),
	               (left_fit, right_fit, y_eval, road_width),
	               (left_rc, left_d, right_rc, right_d)]
	
	# If the detected lines passed the test, record information to the trackers
	if sane(sanity_methods, sanity_args):
		if left_reset_flag: left_lane.reset()
		left_lane.set_detected(True)
		left_lane.add_current_fit_pix(left_fit_pix)
		left_lane.add_current_fit(left_fit)
		left_lane.add_roc(left_rc)
		
		if right_reset_flag: right_lane.reset()
		right_lane.set_detected(True)
		right_lane.add_current_fit_pix(right_fit_pix)
		right_lane.add_current_fit(right_fit)
		right_lane.add_roc(right_rc)
		
		if mid_lane.need_reset(max_failed_count): mid_lane.reset()
		mid_lane.set_detected(True)
		mid_lane.add_current_fit_pix(mid_fit_pix)
		mid_lane.add_current_fit(mid_fit)
		mid_lane.add_roc(mid_rc)
	else:
		# Signal failure of detection to the tracker
		left_lane.set_detected(False)
		right_lane.set_detected(False)
		mid_lane.set_detected(False)

	# Project the lane and add readings the undistorted image
	lane_projected = project_lane(binary_warped, undist, inv_mtx, left_fit_pix, right_fit_pix, margin)

	smoothed_mid_rc = mid_lane.get_smoothed_roc()

	rc_message = rc_text % (int(smoothed_mid_rc))
	ofs_message = ofs_text % (abs(mid_offset), "right" if mid_offset > 0 else "left")
	cv2.putText(lane_projected, rc_message, (50, 49), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,)*3, 2)
	cv2.putText(lane_projected, ofs_message, (50, 99), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,)*3, 2)
	
	return lane_projected


if __name__ == "__main__":

	if len(sys.argv) != 3:
		print("Usage: pipeline.py input_video output")
		sys.exit()

	# Get camera calibration data
	if os.path.isfile(cal_data_name):
		# Load calibration data if data is saved previously
		ret, mtx, dist, rvecs, tvecs = load_camera_data(cal_data_name)
	else:
		# Perform camera calibration
		patterns = glob.glob(cal_dir+cal_names)
		ret, mtx, dist, rvecs, tvecs = calibrate_camera(patterns, pattern_size)
		assert ret, "Camera calibration failed"
		save_camera_data(ret, mtx, dist, rvecs, tvecs, cal_data_name)
	
	
	threshold_methods = [abs_sobel_threshold,
	           S_threshold,
	           R_threshold,
	           H_threshold,
	           SV_threshold]
	
	threshold_args = [('x', 3, (20, 100)),
	        ((170, 230),),
	        ((230, 255),),
	        ((20, 33),),
	        (15, 128)]
	           
	sanity_methods = [roughly_parallel,
	                  properly_separated,
	                  similarly_curved]

	# Calculate perspective transform matrix
	map_mtx = cv2.getPerspectiveTransform(pt_src, pt_dst)
	inv_mtx = cv2.getPerspectiveTransform(pt_dst, pt_src)
	
	# Pipeline Area
	# -------------
	
	input_video = sys.argv[1]
	output_video = sys.argv[2]
	
	clip = VideoFileClip(input_video)
	write_clip = clip.fl_image(process_image)
	write_clip.write_videofile(output_video, audio=False)
	
	



		
