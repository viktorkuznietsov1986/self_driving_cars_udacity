# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/sample_image.png "Example Image"
[image2]: ./examples/dataset_visualized.png "Histogram of Data Distribution"
[image3]: ./examples/augmentation_rotated.png "Rotation"
[image4]: ./examples/do_not_enter.jpg "Do Not Enter"
[image5]: ./examples/construction.jpg "Road Work"
[image6]: ./examples/100kmh.jpg "Speed Limit 100 km/h"
[image7]: ./examples/right_of_way.jpg "Right-of-way"
[image8]: ./examples/turn_right.jpg "Turn right ahead"
[image9]: ./examples/conv3_featuremap.png "Conv3 feature map visualized"
[image10]: ./examples/conv31_featuremap.png "Conv31 feature map visualized"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/self_driving_cars_udacity/CarND-TrafficSigns/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here's how the image looks like:

![alt_text][image1]

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed across the dataset. From the histogram it can be seen that there're at least 200 samples for every traffic sign in the training set.

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step decided to generate additional data because some of the traffic signs have more samples than the others (e.g. 200 vs 2-3k) and that we may end up in a situation where small changes in the input data can drastically affect the performance of the system (e.g. sensitive rotations or scaled images). To do so I utilized the functionality from TensorFlow.

Here's an example of rotated image: 
![alt text][image3]

I used the rotation on a small angle left to right (-15, 15 degrees).
With having that utilized I've got as many as 104397 samples for training set which seems to be huge.

I also explored left-right flipping, but it cannot be used for our data as this leads to improper predictions in the following ways: left turn can be considered as right turn, keep left -> keep right, etc.

As the matter of fact, the tensorflow standard methods made an odd color scheme transformation which led to me having a data set with 2/3 of images being represented in a different color scheme.

As a last step, I normalized the image data because in order to have a smooth learning the image data should have mean zero and equal variance.



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Batch Normalization	|												|
| RELU					|												|
| Convolution 1x1     	| 1x1 stride, same padding, outputs 28x28x6 	|
| Batch Normalization	|												|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 24x24x16 	|
| Batch Normalization	|												|
| RELU					|												|
| Convolution 1x1     	| 1x1 stride, same padding, outputs 24x24x16 	|
| Batch Normalization	|												|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 20x20x32 	|
| Batch Normalization	|												|
| RELU					|												|
| Convolution 1x1     	| 1x1 stride, same padding, outputs 20x20x32 	|
| Batch Normalization	|												|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 10x10x32 				|
| Fully Connected	    | batch norm and dropout, outputs 1000			|
| Fully Connected	    | batch norm and dropout, outputs 500			|
| Softmax				| outputs 43   									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ended up with using AdamOptimizer to minimize the loss function (which is the reduce_mean on softmax_cross_entropy_with_logits).
I used the following hyperparameters during the training:
* Learning Rate: 0.001
* keep_prob for my dropout layers as: 0.5.
* Number of epochs: 50
* Batch size: 128

I experimented with different values for learning rate, keep_prob and number of epochs and ended up with the optimal numbers shown above.
As the future part of my work I plan to get rid of constant number of epochs and come up to early stopping which may help to save training time and get the best training experience.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.982
* test set accuracy of 0.968

I have chosen an iterative approach based on the LeNet architecture.
* First I ran the training on raw LeNet and got the validation set accuracy as 0.89-0.9.
* The model was overfitting and in order to make the things better I did the following:
** Got rid of all the pooling layers except for the last one and replaced them with 1x1 convolutions. This gave me a chance to preserve most of the data during the feed forward.
** Added batch normalization to speed up the learning and reduce the overfitting.
** I tried to go deeper (with 8 convolutions 5x5 followed by 1x1), which given me the validation accuracy around 0.985-0.987, but due to the workspace limitations I got rid of the last 2 convolutional layers and got the validation accuracy as 0.981-0.985.
** I added the dropout layers at the end of fully connected layers to reduce overfitting.

I also plan to look at Inception-based architectures, to get much better validation set accuracy.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because it's part of a bigger image and during image pre-processing (scaling to 32x32) we end up with a silly image of not the best shape. I suppose, I'd better get something with the traffic sign occupying at least half of the image.

The second, 4th and 5th images are pretty easy as they are all wouldn't have any loss during the pre-processing.

The 3rd image will lose a lot of data during scaling and wouldn't be accurate.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Do Not Enter     		| Slippery Road									| 
| Road work    			| Road work										|
| 100 km/h				| Bumpy Road									|
| Right-of-way     		| Right-of-way					 				|
| Turn right ahead		| Turn right ahead     							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. Unfortunately, the 1st and the 3rd images are really tough ones. In order to get them classified properly one need to first cut that part of an image and scale from that point.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a slippery road (probability of 0.52), but the image does contain a No entry sign (it had the probability of 0.075). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .52         			| Slippery road 								| 
| .19     				| Keep right 									|
| .075					| No entry										|
| .059	      			| Priority road					 				|
| .048				    | Road work     								|


For the second image, the model is relatively sure that this is a road work (probability of 0.64), and the image does contain a road work sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .64         			| Road work   									| 
| .346    				| Speed limit (30km/h) 							|
| .005					| Wild animals crossing							|
| .0024	      			| Speed limit (80km/h)			 				|
| .0021				    | Traffic signals								|

For the third image, the model is sure that this is a bumpy road (probability of 0.98), and the image does contain a Speed limit (100 km/h) sign. It doesn't recognize it at all, but this seems to be mostly the scaling issue. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .98         			| Bumpy road   									| 
| .015    				| No vehicles									|
| .001					| Keep right									|
| .001	      			| Yield							 				|
| .0001				    | Priority road									|

For the forth image, the model is relatively sure that this is a Right-of-way at the next intersection sign (probability of 1.00), and the image does contain a Right-of-way at the next intersection sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Right-of-way at the next intersection			| 
| .000    				| Beware of ice/snow							|
| .000					| Pedestrians									|
| .000	      			| Traffic signals				 				|
| .000				    | Double curve									|

For the fifth image, the model is relatively sure that this is a Turn right ahead sign (probability of 1.00), and the image does contain a Turn right ahead sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Turn right ahead								| 
| .000    				| Keep left										|
| .000					| Ahead only									|
| .000	      			| Children crossing				 				|
| .000				    | Go straight or left							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

As the last part of the project I decided to visualize the convolutional feature maps based on the sign "No passing for vehicles over 3.5 metric tons".

The resulting feature maps for the 3rd convolutional layer (5x5) is shown below:
![alt_text][image9]

As shown on the above image, the feature maps are triggering on different aspects of the sign (you can see the full sign on feature maps 26 and 30, as well as different edges and curves which are seems to be specific for other signs).

The resulting feature maps for the following 1x1 convolutional layer is shown below:
![alt_text][image10]

The last layer picks different parts of an input, however it doesn't show any part of full image. It is then being fed to the pooling layer and used as the input for full connected layers.
