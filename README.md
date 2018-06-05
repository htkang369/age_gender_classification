
# Apply CNN models for Gender and Age Classification 

## Overview
This project implements multi-task learning,using CNN model to classify age and gender as well as train age and gender seperately. 
This is my final course project for BigData course in University of Florida. This project focuses on gender and age classification based on images.

The dataset used for training and testing for this project is the IMDB-WIKI - collection of unfiltered face images. It contains total 460,723 images of 20,284 celebrities. There are 2 possible gender labels: M, F. As for age labels, we divided the whole age into 4 intervals, since the celebrities’ characteristic, which means you have to process the raw dataset to create your own labels and data. Each image is labelled with the person’s gender and age-range (out of 4 possible ranges mentioned above). From the original dataset I've used mostly frontal face images reducing the dataset size to 30,000 images. The images are subject to occlusion, blur, reflecting real-world circumstances.
The image link is here: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/


## Preprocessing

The following preprocessing was applied to each image:

Have trained the network on frontal faces images
Create own labels for age and gender.
Wash dataset, there are several whole zero value images, which needed to be cleaned.
Resize each image to 128 x 128.
Transform RGB image to grayscale image.

## Model Description

For Gender Classification, following are the details of the model:

1x7x7 filter shape, 32 feature maps. Stride of 2 and 2 padding. Followed by: ReLU, Max-Pool, LRN
64x5x5 filter shape, 64 feature maps. Followed by: ReLU, Max-Pool, LRN
64x3x3 filter shape, stride 1 and padding 2. ReLU, Max-Pool.
Fully connected layer of 512 neurons. Followed by : ReLU
Fully connected layer of 100 neurons. Followed by : ReLU
Last layer maps to the 2 classes for gender, 4 class for age.
As for multi-task learning, the two tasks share same convolutional layers. After that, they separate to two fully-connected layers. And the parameters could be accessed at our paper and code.

We also applied early-stopping method to avoid over-fitting.
We added L2-regularization terms to avoid over-fitting.
We added two trainable parameters to balance the two tasks so as to push them converge synchronously. 

## Instructions for Running the Model

Ensure the following Python packages are installed on your machine:

opencv
numpy
TensorFlow
scipy
matplotlib
Once your environment has been setup, download the project files and image data and run the following:

For data clean and label generation,execute: python others/wash_data.py
For gender classification execute: python gender/gender.py
For age classification execute: python age/age.py
For multi-task training execute: python joint/jointly.py
For model detection execute: python detect.py

Once you have finish training your model, you could execute: joint/test_model.py or joint/real_time.py to get detection and classification result.

Remember to change path for each input file.

## Results:
Please see the pdf file
