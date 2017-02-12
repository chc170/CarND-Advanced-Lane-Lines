# Advanced Lane Finding Project

## Overview

The goal of this project is to identify lane lines in the video using computer vision techeniques and mark them in the output video. The pipline built for tackling the problem is the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Use color transforms to create a thresholded binary image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./images/image1.png "1"
[image2]: ./images/image2.png "2"
[image3]: ./images/image3.png "3"
[image4]: ./images/image4.png "4"
[image5]: ./images/image5.png "5"
[image6]: ./images/image6.png "6"
[image7]: ./images/image7.png "7"
[image8]: ./images/image8.png "8"
[video1]: ./videos/project_video_output.mp4 "Video"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
### Camera Calibration

#### Calculate Camera Matrix and Distortion Coefficients Using Chessboard Image

Multiple the chessboard pictures taken by the same camera from different angles are prepared in `camera_cal` directory. We will process them to get a camera matrix and distortion coefficients to undistort our images.
In the process I first prepare a (6 x 9) x 3 matrix `obj_pt` to represent a calibration pattern point in 3D space. We will store one of the copy of `obj_pt` to `obj_pts` when chessboard corners are detected in the processed images and store the corners to `img_pts`. The corners are found by `cv2.findChessboardCorners` function. We then feed these two sets of points to `cv2.calibrateCamera` function for calculating our matrix and coefficients. The result will be stored in a pickle file for future use.

![alt text][image1]


### Pipeline (single images)

#### 1. Distortion Correction

With the camera matrix and distortion coefficients from the previous step, we can simply apply the to an image using `cv2.undistort` function.

![alt text][image2]

#### 2. Perspective Transform to Birds-eye View

This is a manual process. I eyeballed two points each of the two lines in a straight lane image, then fixed the x axis positions to the same for points on the same line. Here we will get four new points, which will be the points in the destination image. Use `cv2.getPerspectiveTransform` to get transform matrix then feed the image and the matrix to `cv2.warpPerspective`, we will get a warped image.

![alt text][image3]

#### 3. Thresholded Binary Image

This process is for creating a binary image which tries to set only pixel of interest values to 1. The pixels we are interested are of course the lane lines.

##### Histogram Equalization

To improve the thresholding result, histrogram equalization could be very useful in certain situation (e.g. surface too dark or too bright). This process emphasizes the visual changes in the image. (http://docs.opencv.org/3.2.0/d5/daf/tutorial_py_histogram_equalization.html)

![alt text][image4]

##### Threshold Using Different Color Spaces

In this process, I used 3 different color spaces to threshold the colors. HLS can identify yellow and white by setting different ranges of values. L channel in LUV color space performs well on picking up white color. B channel in LAB color space performs well on identify yellow color. I discarded Sobel graident threshold because the effect is not good.

The following code filter the image by given ranges of each channel.
```
color_min = np.array([channel1_min, channel2_min, channel3_min])
color_max = np.array([channel1_max, channel2_max, channel3_max])
color_mask = cv2.inRange(image, color_min, color_max)
```

![alt text][image5]


#### 4. Find Lane Lines From Binary Image

There are two ways of finding points from binary image. One is starting from scratch. The other is starting with previous found line.

##### Find Points From Scratch

To begin this process, we create a histogram for the bottom half of the image counting the ones in each x value. The index with maximum count on the left half of the histogram will become our starting point of left line. The index with maximum count on the right half of the histogram will become the starting point of right line.
Then, we split the image horizontally into 7 bars and find possible regions of lane lines in each bar. The starting bar is the bottom bar. The next bar we will look for index of maximum count in a range not too far from the previous bar. All the pixels with value 1 in the regions will be used to run a second order polynomial fit. The result looks like the following:

![alt text][image6]

##### Find Points From Previous Found Line

In order to reduce the processing time, we can use the previous found polynomail line and buffer it to create an area that the points of the next line might be in.

![alt text][image7]


#### 5. Draw Boundary and Reversed Perspective Transform

Boundary can be drawn by applying all y values (from 0 to image height) to the fitted polynomail equation to get all the corresponding x values. Reversed perspective transform is simplely reversing the order of the 4 points to create get the matrix. The result will look like the following:

![alt text][image8]

---

### Pipeline (video)

#### Line validation

1. Coefficient comparison: I compare the second order coefficient between frames. The difference should be smaller than 0.0005 according to my experiment. Otherwise, we flag this frame as lane line not detected.
2. Distance between two lines: This is very hard in challenge videos. I can hardly detect consistant lane line distance, so this check is discarded in the final process.
3. Curvature comparison: This has the same issue as 2.,

#### Smoothing

Smoothing is done by averaging last 10 polynomail coefficients if the lines are detected.


Here's a [link to my video result][video1]

---

###Discussion

From my perspective, creating binary image and fitting line are two most difficult and important process. If the threshold performs well, we can easily fit the points to the polynomial, but it is very hard when the environment has much noise like the challenge videos. The noise pushes the difficulties to line fitting process. Small amount of outliers can affect the result very much, so we might have to apply additional technique like RANSAC to ease the affect of outliers. However, most of the computer vision algorithms are not time efficient, so I didn't use them in the project.
