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



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/sethlsx/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library, pickle library and python built-in methods to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

I have a image to show one picture for each label.

![labels]['.\images\labels_display.png']

And here is the histogram for the train data set. It is clear that it's not evenly ditributed.

![histo]['.\images\histogram.png']

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Although the instructions and some blogs on the internet suggests to convert the images to grayscale, I believe the color information is important, as the signs are different in color. I don't wanna lose that information at the beginning.

However, as the image above shown, there are images too dark to be seen clearly. So I used the opencv library to apply equalize hist on the images.

Then I normalize the images.

All these steps are defined in a function called preprocess().

After the preprocessing, the images look like this.


![after_preprocess]['.\images\preprocess.png']

As we can see, the dark images, for example, label 19, 20 etc., can be seen much more clearly.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x43 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 15x15x43 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 13x13x86 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x6x86   				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 4x4x192 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 2x2x192   				|
| Flatten               | outputs 768x1
| Fully connected		| outputs 384x1        							|
| Fully connected		| outputs 192x1        							|
| Fully connected		| outputs 43x1        							|
| logits				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used these hyperparameters:
learning_rate = 0.001
BATCH_SIZE = 3000
EPOCHS = 2000
optimizer = tf.train.AdamOptimizer()

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of 95.8%
* test set accuracy of 94.1%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?


First I tried the original LeNet architecture because I wanna see how it would go, it didn't produce a fine results. Then I decide to add a convolutional full connected layer to compress the images from (32, 32, 3) to (32, 32, 1). I was hoping this could perform as a more "customed grayscale converting" as it transform each pixel in the image using a unique parameter, wheareas in grayscale converting the image is bascally tranfered according to a rather "fixed" rule. However, this didn't produce what I expected.

Then I decide just use the RGB images as input to reserve as much information as possible. Then I decide to raise the depth of the 1st layer and every layer after accordingly. In the convolutional layer, I decide to try the depth of 1st layer as 43 for there are 43 classes. And each layer after is of depth of twice of the depth of the layer before. So 86 for 2nd layer and 192 for 3rd layer. Then in the full connected layer, each layer is  half of the width of the layer before. So 384 for the 1st full connected layer and 192 for the 2nd full connected layer. And in the last layer, it outputs 43 results.(Apologies for my poor english.)

Then I find the network is too complicated to train on my little macbook pro, as it takes 6 hours to finish 200 epochs training. So I decide to set up an aws instances to train the model, which turned out to the smart move. It only takes half an hour to train the model for 2000 epochs and it produced rather satisfying results without having to tune any parameters.

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
I choosed LeNet for the training because that's the only model for now. But I believe this architechture could do so well in this project is because as it does well in the letters classification, this architechture has strong ability to identify curves and shapes in images, which are similar in letters classification and this project.

* Please add some words on what was the process that you ended up with the final model, e.g. what is the process of choosing batch size, number of layers? How do you choose the optimizer and why? What should you do if the model is either overfitting or underfitting?

I choose the batch size 3000 because the instructor said that in his experience that data sets with more than 3000 data points could give reliable results.

As for the number of layers, I start with 3 layers like the original LeNet, then I tried to add a full connected layer as the first layer, it made the model seriously overfitting. So I gave up the idea and went back to 3 layers, after I preprocess the data properly and increase the output depth of the first layer, it produced great results.

I choose the AdamOptimizer simply because it worked pretty well in the LeNet quiz. However, as reviewer raised the question, I did some research. According to this [paper](https://arxiv.org/pdf/1609.04747.pdf), Adam is a better optimizer because it usually converges much more quickly than other optimizer in most scenarios.

My model do have overfitting problems as the accuracy on train data set is higher than validation set and test set. To further improve the model, I would decrease the first layer depth from 43 to 30s as I can't decrease keep probility anymore.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text]['images\images.jpeg'] ![alt text]['images\images-2.jpg'] ![alt text]['images\images-3.jpg'] 
![alt text]['images\images-4.jpeg'] ![alt text]['images\images-5.jpg']


The images I found are basically taken from a front angle, are fairly bright, of high contrast. There are little jitteriness either. But there are backgound objects in all of them, especially the 1st and 4th images.The first and fourth images might be difficult to classify because there are lots of trees, buildings and cars in the background which could form false lines for the model to detect.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			                        |     Prediction	        					| 
|:-------------------------------------:|:---------------------------------------------:| 
| Stop Sign      		                | Stop sign   									| 
| Road work     			            | Road work 									|
| Speed limit(70km/h)					| Speed limit(70km/h)							|
| No entry	      		                | No entry					 				    |
| Right-of-way at the next intersection	| Right-of-way at the next intersection      	|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.1%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is absolutely sure that this is a stop sign (probability of 1.00), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Stop sign   									| 
| .00     				| Speed limit (20km/h) 							|
| .00					| Speed limit (30km/h)							|
| .00	      			| Speed limit (50km/h)			 				|
| .00				    | Speed limit (60km/h)							|


For the second image, the model is absolutely sure that this is a Road work sign (probability of 1.00), and the image does contain a Road work sign. The top five soft max probabilities were
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Road work   									| 
| .00     				| Bumpy road        							|
| .00					| Children crossing  							|
| .00	      			| Right-of-way at the next intersection			|
| .00				    | Road narrows on the right						|

For the third image, the model is relatively sure that this is a Speed limit (70km/h) sign (probability of .97), and the image does contain a Speed limit (70km/h) sign. The top five soft max probabilities were
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .97         			| Speed limit (70km/h)							| 
| .03     				| Speed limit (30km/h)  						|
| .00					| Speed limit (20km/h)							|
| .00	      			| Speed limit (120km/h)             			|
| .00				    | Speed limit (100km/h) 						|

For the fourth image, the model is absolutely sure that this is a No entry sign (probability of 1.00), and the image does contain a No entry sign. The top five soft max probabilities were
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| No entry   									| 
| .00     				| Speed limit (20km/h)       					|
| .00					| Speed limit (30km/h)							|
| .00	      			| Speed limit (50km/h)			                |
| .00				    | Speed limit (60km/h)  						|

For the second image, the model is absolutely sure that this is a Right-of-way at the next intersection sign (probability of 1.00), and the image does contain a Right-of-way at the next intersection sign. The top five soft max probabilities were
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Right-of-way at the next intersection   		| 
| .00     				| Speed limit (20km/h)   						|
| .00					| Speed limit (30km/h) 							|
| .00	      			| Speed limit (50km/h)              			|
| .00				    | Speed limit (60km/h)	     					|

The actual outputs of top 5 softmax is as follows:

TopKV2(values=array([[1.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00],
       [1.0000000e+00, 5.4256552e-16, 7.5246122e-18, 8.0823714e-19,
        1.2141358e-19],
       [9.7361767e-01, 2.6382184e-02, 1.1637754e-07, 2.2291743e-10,
        2.0435522e-12],
       [1.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00],
       [1.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00]], dtype=float32), indices=array([[14,  0,  1,  2,  3],
       [25, 22, 28, 11, 24],
       [ 4,  1,  0,  8,  7],
       [17,  0,  1,  2,  3],
       [11,  0,  1,  2,  3]], dtype=int32))
### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


