## Project Writeup

---

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

[image1]: ./calibration_example.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./distortion_correction_testimage.png "Distortion Corrected Road"
[image4]: ./threshold_pipeline_result.png "Binary Example"
[image5]: ./warped_straight_lines.png "Warp Example"
[image6]: ./color_fit_lines.png "Fit Visual"
[image7]: ./output_images/test1.jpg "Output"
[video1]: ./project_video_processed.mp4 "Project Video"
[video2]: ./challenge_video_processed.mp4 "Challenge Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in "P2.ipynb". Please see the class CameraCalibration.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

I instantiate the class CameraCalibration and call the method `camera.undistort`. The resulting image is as follows:
![alt_text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

In order to create a binary image, containing likely lane pixels I implemnted a method apply_threshold which does the following:
    - Converts an image in BGR color space to HLS and retrieve the S channel. This is needed because the S channel in HLS preserves pretty much of the shape given different lighting conditions.
    - Apply Sobel X and Y filters to S and L channels.
    - Threshold X and Y gradients on S and L channel (I used threshold values (50, 200)).
    - Threshold H, L, S channels to identify white and yellow colors:
        - yellow - H (0,85), L (120,255), S (75,255)
        - white - H (20,255), L (170,255), S (0,80)
    - Combined binaries from thresholds which gives a pretty accurate lane lines.
    
The above values were tuned by using the ipywidgets. See the corresponding section in the notebook.

The resulting image looks as follows:
![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Now it's time to adjust the pipeline, so it uses the Perspective Transform to convert the images to birds eye view. This is needed to further process an image in order to find the lane lines. The transformation can be done as follows:

    - define the corresponding points on source and destination images to calculate the transformation matrix.
    - calculate the transformation matrix using cv2.getPerspectiveTransform
    - apply the transformation by using cv2.warpPerspective


```python
src = np.float32([[1.8*img_size[0]/6,img_size[1]], [2.9*img_size[0]/6,1.9*img_size[1]/3], [3.4*img_size[0]/6,1.9*img_size[1]/3], [img_size[0]-offset//2,img_size[1]]])
    dst = np.float32([[1.5*img_size[0]/6,img_size[1]], [1.5*img_size[0]/6,0], [img_size[0]-offset,0], [img_size[0]-offset,img_size[1]]])
```

The resulting image is shown below. As you can see, the lane lines appear to be parallel on the transformed image.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I used the sliding windows technique to fir the lane lines with 2nd degree polynomials. To do so I followed the steps below:
 - Create a histogram of the bottom half of the image to actually find the points where to start searching for a lane lines. The lane lines starting points can be easily identified as the peaks on in the left and right parts of histogram (1 on each side).
 - Use the sliding window to fit lane lines polynomials having the starting points identified form a histogram. The resulting binary warped image with lane lines identified is shown below as well as the source code to achieve that.
 
See method `find_lane_pixels` in `P2.ipynb` for implementation details.
The resulting image will look like this:

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
First of all, as all the distances on the images are represented as pixels, I needed to implement the conversion from pixels to meters on both axes. We define that there's the following conversion ratios for `y` and `x` dimensions:
```python
    ym_per_pix = 30/img_shape[1] # meters per pixel in y dimension
    xm_per_pix = 3.7/img_shape[0] # meters per pixel in x dimension
```
The conversion is implemented in method `get_meters_per_pix`.

Next, to measure image curvature, we need to calculate the curvature for both lane lines based on the polynomial coefficients in the following way:
```python
    left_curverad = ((1+(2*left_fit_cr[0]*y+left_fit_cr[1])**2)**(3/2))/np.abs(2*left_fit_cr[0]) 
    right_curverad = ((1+(2*right_fit_cr[0]*y+right_fit_cr[1])**2)**(3/2))/np.abs(2*right_fit_cr[0]) 
```

The distance from center is calculated as follows:
 - Find lane pixels
 - Fit a second order polynomial by feeding the points converted to meters
 - Get the 'x' coordinate for the center of an image (the position of the camera) and 'x' position for the center of the lane.
 - Substract 'x' corresponding to the center of an image from the 'x' corresponding to the center of the lane.

Please see the methods `measure_image_curvature` and `get_distance_from_center` in `P2.ipynb`.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Let's implement the pipeline to process single image. It will contain the following steps:

    - Preprocess image: undistort, apply color threshold and apply the perspective transform.
    - Find lane lines, by finding the grade 2 polynomial coefficients corresponding to the lane lines.
    - Calculate lane curvature and position of the car in relation to the center of the lane.
    - Draw the polygon occupying the area between the identified lane lines.
    - Combine the original image and the empty image with lane lines drawn.
    - Write a text on the resulting image telling the info about lane curvature and distance from the center of the lane.

You can find the implementation in the method `process_single_image` in `P2.ipynb`.

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

- Here's a link on project video [video1](./project_video_processed.mp4)

- Here's a link on challenge video [video2](./challenge_video_processed.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline coded above consists of the following steps (you can review the results in 'project_video_processed.mp4' and 'challenge_video_processed.mp4'):

    - Preprocess image: undistort, apply color threshold and apply the perspective transform.
    - Find lane lines, by finding the grade 2 polynomial coefficients corresponding to the lane lines.
        - If it's the first frame, or after the number of unsuccessful frames, use histogram to find the starting position for sliding window to find the lane lines.
        - If the previous frame was successfully processed, use the averaged polynomial coefficients for several previous frames to look around these lines +- margin.
            - Run sanity check and do the smoothing
                - Sanity check: if the coefficients differ by the value .1, grab the previous coefficients, say that the lane lines were not found the way they should and work out this way.
                - Smoothing: in order to smooth the lane lines on the video I was getting the new coefficients which are the result of the following summation: .95 old_coeff + .05 new_coeff
    - Calculate lane curvature and position of the car in relation to the center of the lane.
    - Draw the polygon occupying the area between the identified lane lines.
    - Combine the original image and the empty image with lane lines drawn.
    - Draw a text on the resulting image telling the info about lane curvature and distance from the center of the lane.

So, my pipeline for the video processing does a good job on the main project video file and does an average job on challenge video file. The thresholding technique can be improved to work better under different lighting conditions (it works well for most of the cases though). However, if the curvature of the lane line frequently changes on large values, it wouldn't work well as the sanity check combined with smooting technique will prevent it from drawing the accurate lane lines. In order to make a good progress with finding any lane lines, it's worth considering to use the machine learning techniques, e.g. semantic segmentation which may give the opportunity to actually build the robust solution to accurately find and smoothen the lane lines. I plan to work towards this in my spare time to actually get this up and running.
