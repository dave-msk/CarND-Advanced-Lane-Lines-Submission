import numpy as np
from collections import deque

# Define a class to receive the characteristics of each line detection
class Line(object):
	def __init__(self, n_cache):
		# maximum number of records to be maintained
		self.n_cache = n_cache
		# was the line detected in the last iteration?
		self.detected = False
		# number of times detection failed successfully
		self.failed_count = float("inf")
		# polynomial coefficients for the most recent fit
		self.current_fit = deque()
		self.current_fit.append(np.array([False]))
		
		self.current_fit_pix = deque()
		self.current_fit_pix.append(np.array([False]))
		# radius of curvature of the line in some units
		self.radius_of_curvature = deque()
	
	def add_current_fit(self, lane_fit):
		self.current_fit.append(lane_fit)
		self._maintain(self.current_fit)
	
	def add_current_fit_pix(self, lane_fit):
		self.current_fit_pix.append(lane_fit)
		self._maintain(self.current_fit)
	
	def add_roc(self, roc):
		self.radius_of_curvature.append(roc)
		self._maintain(self.radius_of_curvature)
	
	def set_detected(self, detected):
		self.detected = detected
		if detected: self.failed_count = 0
		else: self.failed_count += 1
	
	def get_last_fit(self):
		return self.current_fit[-1]
	
	def get_last_fit_pix(self):
		return self.current_fit_pix[-1]
	
	def get_smoothed_fit(self):
		if self.current_fit:
			return np.mean(self.current_fit, axis=0)
		return np.array([0,0,0], dtype='float')
	
	def get_smoothed_roc(self):
		if self.radius_of_curvature:
			return np.mean(self.radius_of_curvature)
		return 0
	
	def need_reset(self, max_failed_count):
		if self.detected: return False
		if self.failed_count < max_failed_count: return False
		return True
	
	def reset(self):
		self.current_fit.clear()
		self.radius_of_curvature.clear()
	
	def _maintain(self, dq):
		while len(dq) > self.n_cache:
			dq.popleft()
	
	
