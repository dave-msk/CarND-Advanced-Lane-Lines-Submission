import numpy as np


def sane(methods, args):
	"""
	Performs a series of sanity checks.
	
	Arguments
	---------
	    methods: List of sanity check methods
	       args: List of corresponding arguments
	
	Returns: True if all tests are passed, otherwise False.
	"""
	if len(methods) != len(args): return False
	for i in range(len(methods)):
		if not methods[i](*args[i]): return False
	return True


def roughly_parallel(left_fit, right_fit, y_eval, threshold=np.pi/36):
	"""
	Check if the two fitted curves are parallel. This is done by
	taking two points from each curves and detemine whether the
	straight lines obtained share roughly the same elevation angle.
	
	Arguments
	---------
	    left_fit: Polynomial fit of the left lane line
	   right_fit: Polynomial fit of the right lane line
	      y_eval: y-coordinate of the evaluation point other than y=0
	   threshold: Allowed difference of elevation angle to declare parallelism
	
	Returns: True if difference is below threshold, False otherwise
	"""
	left_dx = left_fit[0]*y_eval**2 + left_fit[1]*y_eval
	left_elev = np.arctan2(y_eval, left_dx)
	
	right_dx = right_fit[0]*y_eval**2 + right_fit[1]*y_eval
	right_elev = np.arctan2(y_eval, right_dx)
	
	return True if (abs(left_elev - right_elev) < threshold) else False


def properly_separated(left_fit, right_fit, y_eval, exp_dist, threshold=0.2):
	"""
	Checks if the given polynomials are properly separated.
	
	Arguments
	---------
	    left_fit: Polynomial fit of the left lane line
	   right_fit: Polynomial fit of the right lane line
	      y_eval: y-coordinate of the evaluation point
	    exp_dist: Expected distance of separation
	   threshold: Allowance of difference of distances
	"""
	leftx = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
	rightx = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
	return abs(abs(rightx - leftx) - exp_dist) < threshold


def similarly_curved(left_rc, left_d, right_rc, right_d, threshold=2):
	"""
	Determines if the curvatures are similar by comparing the segment area.
	
	Arguments
	---------
	      left_rc: Radius of curvature of the left lane line
	       left_d: Length of chord of the left lane line
	     right_rc: Radius of curvature of the right lane line
	      right_d: Length of chord of the right lane line
	    threshold: Allowance of difference of segment area
	"""
	left_theta = np.arccos(1 - (left_d/left_rc)**2/2)
	left_seg = (left_theta-np.sin(left_theta))*left_rc**2/2
	
	right_theta = np.arccos(1 - (right_d/right_rc)**2/2)
	right_seg = (right_theta-np.sin(right_theta))*right_rc**2/2
	
	return abs(left_seg-right_seg) < threshold
