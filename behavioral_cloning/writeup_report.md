# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center.jpg "Center Lane Driving (Central camera)"
[image2]: ./examples/left.jpg "Center Lane Driving (Left camera)"
[image3]: ./examples/train_validation_loss.JPG "Changes In the Validation Loss"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* run1.mp4 containing the video record of autonomous driving

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes, depths between 24 and 64 (model.py lines 66-101) 

The model includes RELU layers to introduce nonlinearity (code lines 68, 72, 76, 80, 84, 92, 96, 100), and the data is cropped using a Keras cropping layer and normalized in the model using a Keras lambda layer (code lines 62, 63). The model also employes the batch normalization to make the training faster and reduce overfitting. 

#### 2. Attempts to reduce overfitting in the model

The model contains batch normalization layers in order to reduce overfitting (model.py lines 67, 71, 75, 79, 83, 90, 94, 98). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 16 and see the implementation of method generator()). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 104).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving in both directions and employ the data from left, right and center cameras. I used the correction coefficient to be added / subtracted from the angle to be used for left and right camera images. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the architecture proposed by NVidia team with adding batch normalization layers to reduce overfitting and speed up the training.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 
I also augmented the steering data in order to use the images from left and right cameras for driving recovery.
It turned out that training experience with my initial and final CNN architecture was painful as running 1 full epoch with batch size 32 has taken more than 2 hours.
I've added the dropout layers to my architecture first, but due to the low number of epochs it was not helpful and was even bad during the inference time.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track (there's the part of the road where it's hardly seen the right lane line, so after epoch 1 the car was not considering it this way and was going off the track). To improve this, I was forced to turn off the dropout layers (as it's helpful if you train the model for large number of epochs). I also increased the number of training epochs to 3 (I couldn't increase more as each time I leave the browser, there's a risk the workspace goes off which stops the training).

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 59-106) consisted of a convolution neural network with the following layers and layer sizes:
* Cropping2D to crop the input image
* Lambda to do the normalization
* Conv2D with 5x5x24, stride 1, padding='SAME', batch normalization and ReLU activation function
* MaxPooling2D with pool size 2x2 and strides 2
* Conv2D with 5x5x36, stride 1, padding='SAME', batch normalization and ReLU activation function
* MaxPooling2D with pool size 2x2 and strides 2
* Conv2D with 5x5x48, stride 1, padding='SAME', batch normalization and ReLU activation function
* MaxPooling2D with pool size 2x2 and strides 2
* Conv2D with 3x3x64, stride 1, padding='SAME', batch normalization and ReLU activation function
* MaxPooling2D with pool size 2x2 and strides 2
* Conv2D with 3x3x64, stride 1, padding='SAME', batch normalization and ReLU activation function
* MaxPooling2D with pool size 2x2 and strides 2
* Fully connected layer with 1164 neurons, batch normalization and ReLU activation function
* Fully connected layer with 100 neurons, batch normalization and ReLU activation function
* Fully connected layer with 50 neurons, batch normalization and ReLU activation function
* Output layer with 1 neuron, no activation


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

![alt_text][image2]

I then recorded 2 laps on track one using center lane driving in opposite direction.

In order to reproduce the lane position recovery I used the images captured by the left and right cameras.

After the collection process, I had about 18000 of data points. I then preprocessed this data by cropping and normalizing the data. As I was doing it on the go, it appeared to be not a good idea as it was making the epoch go longer.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 due to high computational cost of running the training on the GPU workspace. I used an adam optimizer so that manually training the learning rate wasn't necessary.

The changes in the loss can be seen here:

![alt_text][image3]

