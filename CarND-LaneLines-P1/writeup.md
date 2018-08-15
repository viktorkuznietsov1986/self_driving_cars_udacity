# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. The pipeline for finding lane lines and drawing them on image / video frames.

My pipeline consisted of the following steps:
- I converted the image to RGB and then converted to grayscale.
- Applied GaussianBlur to smoothen the grey image.
- Used Canny Edge detection to find all the edges given the threshold.
- Chosen the region of interest where I will be looking for the lane lines and applied that mask to the result from Canny Edge detection. I got the region of polygon shape to look at.
- Applied the hough transform to get these lines. I used standard values for rho and theta (1 and np.pi/180 respectively). The threshold was set to 60 to get rid of noise. As the result I've got multiple lines corresponding to the lane lines (lane lines have some gaps in between).
- Combined the initial image and the image with lines to get the final image with lane markins.

In order to draw a single line on the left and right lanes, I modified the draw_lines() by doing the approximation of slope and y-intercept for the lines found. This way, I'm getting one slope/y-intercept for left line and one slope/y-intercept for the right one. Having y-bounds defined, it's straightforward to calculate the corresponding x values.

This works perfectly fine with images. If you run the pipeline, you can find the resulting images in the specfied folder.

Switching to the video processing, the things are getting more complex. The slope/intercept are changing between frames, so the lane lines should be drawn with a very smooth change. In order to do so, I introduced the caching of the slope/y-intercept, so the next values for slope/y-intercept are actually the average between the specific number of previous ones and the current one. 

I introduced the class LaneFinder which actually solves both image and video processing tasks. If it is to be used with a single images, the caching must be disabled. If we are talking about video processing, one must enable the caching and set up the proper values for delta (the maximum difference between current slope value and the averaged slope value where we actually accept the current slope value) and number_of_slopes which is the number of frames to look in the past to get the average value.



### 2. Potential shortcomings current pipeline

One potential shortcoming would be what would happen when the road has many turns which means large changes in slope/intercept. 

Another shortcoming could be if there's a vehicle just in front of our car which actually hides the part of lane lines. This level of noise may make the identification hard enough.


### 3. Possible improvements to my pipeline

A possible improvement would be to adjust the area where we try to identify the lane lines. It can be a single polygon or several polygones if the lanes have lots of turns (right now it's hardcoded). This can be done when we have the changes in slope/y-intercept for the single line in one image. A different caching needs to be used there to actually rely on this. Right now the area is a single polygon with hard-coded vertices which are actually depend on the dimensions of an image.
