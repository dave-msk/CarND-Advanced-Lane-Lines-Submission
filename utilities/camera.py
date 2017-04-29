import numpy as np
import pickle
import cv2


def load_camera_data(data_file):
	"""
	Loads camera calibration data from pickled data
	
	Arguments
	---------
	data_file: pickled camera data file
	
	Returns: ret, mtx, dist, rvecs, tvecs (same return as cv2.calibrateCamera)
	"""
	with open(data_file, 'rb') as f:
		data = pickle.load(f)
	mtx = data['mtx']
	dist = data['dist']
	rvecs = data['rvecs'] # not used
	tvecs = data['tvecs'] # not used
	ret = True
	return ret, mtx, dist, rvecs, tvecs


def save_camera_data(ret, mtx, dist, rvecs, tvecs, data_file):
	"""
	Pickles camera calibration data into file.
	
	Arguments:
	ret, mtx, dist, rvecs, tvecs: Outputs from cv2.calibrateCamera
	                   data_file: Output file name
	"""
	data = {'mtx': mtx,
		   'dist': dist,
		  'rvecs': rvecs,
		  'tvecs': tvecs}
	pickle.dump(data, open(data_file, "wb"))


def calibrate_camera(patterns, patternSize):
	"""
	Performs camera calibration
	
	Arguments
	---------
	       patterns: list of image filenames of chessboards with the same pattern size
	    patternSize: pattern size in the format of tuple
	               - (points_per_row, points_per_column)
	
	Returns: (retval, cameraMatrix, distCoeffs, rvecs, tvecs)
	
	          retval: return value that indicates success or failure
	    cameraMatrix: (intrinsic) camera matrix
	      distCoeffs: distortion coefficients
	           rvecs: rotation vectors (extrinsic)
	           tvecs: translation vectors (extrinsic)
	"""
	objp = []
	imgp = []
	grid = np.zeros((patternSize[0]*patternSize[1], 3), np.float32)
	grid[:,:2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1,2)
	
	for pattern in patterns:
		img = cv2.imread(pattern)
		if len(img.shape) > 1:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		ret, corners = cv2.findChessboardCorners(img, patternSize, None)
		if ret:
			objp.append(grid)
			imgp.append(corners)
	
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objp,
	                                                   imgp,
	                                                   img.shape[::-1],
	                                                   None,
	                                                   None)
	return ret, mtx, dist, rvecs, tvecs
