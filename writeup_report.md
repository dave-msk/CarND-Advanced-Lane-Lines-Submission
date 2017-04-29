**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[undistorted_chessboard]: output_images/undistorted_chessboard.jpg
[undistorted_example]: output_images/ex_undist_example.jpg
[sobel_thresholding]: output_images/ex_binary_x.jpg
[s_thresholding]: output_images/ex_binary_s.jpg
[r_thresholding]: output_images/ex_binary_r.jpg
[h_thresholding]: output_images/ex_binary_y.jpg
[sv_thresholding]: output_images/ex_binary_w.jpg
[combined]: output_images/ex_binary.jpg
[warped]: output_images/ex_warped_example.jpg
[binary_warped]: output_images/ex_binary_warped.jpg
[lane_detected]: output_images/ex_fitted.jpg
[formula]: output_images/formula.png
[lane_projected]: output_images/ex_projected.jpg
[projected_with_text]: output_images/ex_output.jpg
[approach_x]: output_images/binary_x.jpg
[approach_s]: output_images/binary_s.jpg
[approach_r]: output_images/binary_r.jpg
[approach_y]: output_images/binary_y.jpg
[approach_w]: output_images/binary_w.jpg
[approach_failed_y]: output_images/failed_y.jpg
[approach_failed_mag]: output_images/failed_mag.jpg
[approach_failed_dir]: output_images/failed_dir.jpg
[approach_failed_gray]: output_images/failed_gray.jpg

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

**NOTE**: Other than `pipeline.py`, all code files are located in the `utilities` directory.
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

In this step, a function `calibrate_camera` is defined to perform the camera matrix and distortion coefficients computation process. The code is located at lines 41 through 78 in `camera.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `grid` is just a replicated array of coordinates, and `objp` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgp` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objp` and `imgp` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Undistorted Images][undistorted_chessboard]

### Pipeline (single images)

The global data used in the pipeline, such as calibration data (camera matrix and distortion coefficients) and hyperparameters, is first computed/loaded in the program (lines 380-416 in `pipeline.py`) before the pipeline process begins.

#### 1. Provide an example of a distortion-corrected image.

Here is an example of an image together with its undistorted one.
![Original and Undistorted][undistorted_example]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image.

Five thresholding approaches are employed. The parameters are defined at lines 400 through 410 in `pipeline.py`. The code for the following thresholding processes are defined in `thresholding.py`.

1. Sobel thresholding in x-direction (lines 5-28) with 3-by-3 kernel size and thresholds (20, 100).

    ![Sobel Thresholding][sobel_thresholding]

2. S-Channel thresholding in HLS color space (lines 86-97) with thresholds (170, 230).

    ![S-Channel Thresholding][s_thresholding]

3. R-Channel thresholding in RGB color space (lines 100-110) with thresholds (230, 255).

    ![R-Channel Thresholding][r_thresholding]

4. H-Channel thresholding in HSV color space (lines 113-124) with thresholds (20, 33) to find if there is any yellow line.

    ![H-Channel Thresholding][h_thresholding]

5. SV-Channel thresholding in HSV color space (lines 127-142) with `s <= 15` and `v >= 128` to find if there is any white line.

    ![SV-Channel Thresholding][sv_thresholding]

The above five thresholded binary images are then combined to give the following result:

![Combined Binary Image][combined]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is included in the main `process_image()` function (line 292). The perspective transform matrix is preprocessed before the pipeline process begins (line 415). I chose to hardcode the source and destination points (lines 28-38 in `pipeline.py`):

| Source        | Destination   | 
|:-------------:|:-------------:| 
|  202, 720     |  250, 720     | 
| 1103, 720     | 1030, 720     |
|  701, 460     | 1030, 0       |
|  580, 460     |  250, 0       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Warped Example][warped]

The warped binary image from the above example is as follows:

![Warped Binary Image][binary_warped]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Two approaches are employed. If the the lane lines are not detected three times in a row, the warped binary image from the previous stage will go through a sliding window search process (`sliding_window_search()` at lines 60-126 in `pipeline.py`) to try to locate the lane lines. Otherview, the previously detected lane line is used to detect the current lane lines (`fine_tune_search()` at lines 128-156 in `pipeline.py`) as their positions would not differ significantly. The following is an illustration of detected lane lines in the warped binary image in the example.

![Warped Binary Image with Lane Detected][lane_detected]

In both approaches, the functions return the white pixel locations in the warped binary image that is close to the potential lane lines separately. These candidate pixels are then used to fit a 2nd-order polynomial using `polyfit()` (at lines 159-177 in `pipeline.py`).

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is calculated by `radius_curvature()` and the position of the vehicle with respect to center is computed by `offset()` at lines 180-191 , and  194-205 in `pipeline.py` respectively.

`radius_curvature()` accepts an array of coefficients of a polynomial and an evaluation point (which is the bottom of image in this case) and computes the radius of curvature using the formula:

![Radius of Curvature Formula][formula]

Similarly, `offset()` accepts an array of coefficients of a polynomial, an evaluation point, and this time also the x-coordinate of image center for comparison.

##### Sanity Check

The polynomials obtained would go through a sanity check (line 343 in `pipeline.py`) to ensure they give reasonable results. The sanity check focuses on three aspects:

1. Parallelism
2. Separation of Lane Lines
3. Similarity of Curvature

The code is included in `sanity_check.py` and the description of each approach will be discussed in the `Discussion` section.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 208 through 267 in my code in `pipeline.py` in the function `project_lane()`. Here is an example of my result on a test image:

![Undistorted Image with Lane Projected][lane_projected]

Here is the same image with radius of curvature and offset included.

![Undistorted Image with Lane Projected and Information Included][projected_with_text]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

The first thing I did was to build a code base for fast prototyping.

The first version of the code performs thresholding, lane line finding and computes radius of curvature and offset to the eight given test images.

The second version of the code returns a modified video in which each frame is processed. I used the project video to discover possible factors why the pipeline fails to identify lane lines in different cases.

##### Pseudo-Code of Pipeline:

- Input: Distorted image from camera (`image`)
- Output: Image with lane projected and information included

1. `undist` <-- undistort(`image`)
2. `binary` <-- thresholding_pipeline(`undist`)
3. `binary_warped` <-- warped_image_to_birdeye_view(`binary`)
4. `lanes` <-- search_lane(`binary_warped`)
5. `metrics` <-- calculate_metrics(`lanes`)
6. If is_sane(`lanes`, `metrics`): `tracker`.track_data(`lanes`, `metrics`)
7. Else: `tracker`.track_failure()
8. `result` <-- draw_lane_and_info_to_image(`undist`, `tracker`)
9. Return `result`

The following describes each thresholding method I tried and the reason why it is included in the pipeline or not. 

##### Threshlding Approaches

The main selection criteria for choosing a thresholding approach is that whether it:

- Preserves lane lines, and
- Filters out noises on the road.

The first criterion ensures there is information for the line detection algorithm to work with. The second criterion helps to reduce distraction, so that the algorithm will not be misled seriously.

The hyperparameters are obtained by trial and error. If an approach is not chosen, the worst case of the best try would be illustrated.

1. Sobel Thresholding in x-direction (Selected):
    
    ![Sobel Thresholding in x-direction][approach_x]
    
    - This approach is selected as we can see it gives a pair of fairly clear lane lines, while the noise on the road are not quite heavy.
    - Video with each frame threholded goes [here](./videos/project_out_x.mp4).

2. Sobel Thresholding in y-direction (Not Selected):

    ![Sobel Thresholding in y-direction][approach_failed_y]

    - This approach is not selected as it is either noisy or it gives no useful information on where the lane lines are.

3. Thresholding by Magnitude of Gradient (Not Selected):

    ![Thresholding by Magnitue of Gradient][approach_failed_mag]

    - This approach is like the previous one. The amount of noise is fairly equal to the amount of lane line information. Therefore this approach is not included in the pipeline.

4. Thresholding in Gradient Direction (Not Selected):

    ![Thresholding in Gradient Direction][approach_failed_dir]

    - This approach introduces a significant amount of noise. It would not be suitable to apply this method to the raw image directly. This is further discussed in the coming section.

5. Thresholding in R-Channel (Selected):

    ![Thresholding in R-Channel][approach_r]

    - As we can see, this approach is fairly robust, and it give a significantly clear image of lane lines none to negligible noise. This is one of the main contributors in the thresholding process in the pipeline.
    - Video with each frame threholded goes [here](./videos/project_out_r.mp4).

6. Thresholding in S-Channel (Selected):

    ![Thresholding in S-Channel][approach_s]

    - The thresholded images are also quite robust in this approach. The main reason why full saturation is filtered out is that color changes on the road by external factors, such as shadow, would have a huge effect on its saturation. Therefore only a narrow part of the saturation continuum is used.
    - At around 0:40 of this [video](./videos/project_out_s.mp4) shows this problem.

7. Thresholding in H-Channel (Selected):
    
    ![Thresholding in H-Channel][approach_y]

    - This approach (and also the next one) is inspired by the fact that the lane lines are either yellow or white (and therefore no artificially made road should be in yellow or white, which would otherwise make those lines indistinguishable even for human). This method detects the pixels fell in the yellow range of the hue in HSV color space. As we can see it works reasonable well and the noise are not heavy.
    - Video with each frame threholded goes [here](./videos/project_out_y.mp4).

8. Thresholding in SV-Channel (Selected):

    ![Thresholding in SV-Channel][approach_w]

    - This approach is the white counterpart of the above. To find the pixels close to white, we require, in HSV, the saturation to be low and value to be reasonably high (otherwise it would be close to black rather than white).
    - Video with each frame threholded goes [here](./videos/project_out_w.mp4).

9. Thresholding on Grayscaled Image (Not Selected)s:

    ![Thresholding on Grayscaled Image][approach_failed_gray]

    - This approach loses too much information in the processing of grayscale conversion. One can see that the left line appears and disappears together with the noise on the road.


To conclude, The selected approaches are: Sobel Thresholding in x-direction, S-Channel, H-Channel, SV-Channel, and R-Channel.

I have included a video to show the [thresholded output](./videos/project_out_binary.mp4) on the project video and its [perspective transformed](./videos/project_out_binary_warped.mp4) version.

##### Sanity Check

The lane lines discovered by the detection algorithms (`sliding_window_search()` and `fine_tune_search()` in `pipeline.py`) will go through a sanity check to see whether the findings make sense before they are used in further processing. The code is included in `sanity_chech.py` and it consists of three parts:

1. Parallelism

     This part (`roughly_parallel()` at lines 21-42) checks whether the detected lane lines are roughly parallel. The check is done by comparing whether the inclination angle between the x-axis and the left lane chord is close to that of the right lane chord.
    
2. Separation of Lane Lines

    This part (`properly_separated()` at lines 45-59) compares the separation of the lane lines at a particular point (the bottom of the image is chosen) is close to the expected lane width.

3. Similarity of Curvature

    This part (`similarly_curved()` at lines 62-80) decides whether the curvatures found are similar to each other.
    
    This is a particularly tricky task as one has to take care of the fact that straight lines are similarly curved, but their radius of curvature can differ significantly as their theoretical value is the infinity. Therefore both the absolute and relative value comparison do not work.

    This problem is addressed by comparing the area of the segment between the arc and the chord of the curve defined by the lane line polynomial instead. This approach comes from the fact that the more the lane line curved, the larger segment area it introduces. So even if the radius of curvature of left and right lane line differ significantly, if both of them roughly represents a straight line, the difference in their segment areas should stay negligible.

##### Lane Tracking

At each frame, the lane data is tracked by the `Line` class defined in `line.py`. This class tracks the most recently succeeded lane line detection results and data. If the lane line data did not pass the sanity test, the failure would be recorded and the pipeline would be forced to detect the lane lines from scratch again (using `sliding_window_search()`) when the maximum number of consecutive failures is achieved.

The tracker also performs smoothing on the data found, namely radius of curvature and offsets. This would mitigate the rapid fluctuation phenomonen of data at the expense of slight delay in the information about the vehicle's current state.

##### Drawbacks and Potential Modifications

One of the main problems of the pipeline introduced above is that all the parameters are hardcoded. This means that if the environment is rather bright or dim, or for some reason the road is not uniformly colored (possibly due to roadworks), this pipeline would fail to isolate the lane lines from the scene. The followings are several potential solutions:

1. Use dynamic thresholds for lane lines extraction according to the statistics of pixel value distribution, such as mean and standard deviation of intensities in channels.

2. Perform connected component analysis to find objects connected in terms of color, then extract objects with a thin shape. The extraction can be done by fitting a polynomial and determine whether over 90% of the pixels of the object are within a margin from the fitted curve.