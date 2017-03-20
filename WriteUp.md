##Writeup
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car_not_car]: ./output_images/car_not_car.png
[Car_HOG]: ./output_images/car_and_car_HOG.png
[Car-CH-1-Color-histogram]: ./output_images/Car-CH-1-Color-histogram.png
[Car-CH-1-Spatial-binning]: ./output_images/Car-CH-1-Spatial-binning.png
[not-Car_CH-1_and_HOG]: ./output_images/not-Car_CH-1_and_HOG.png
[not-Car_CH-1_Color_histogram]: ./output_images/not-Car_CH-1_Color_histogram.png
[not-Car_CH-1_Spatial_inning]: ./output_images/not-Car_CH-1_Spatial_inning.png
[slide_window]: ./output_images/slide_window.png
[example]: ./output_images/example.png
[video_img]: ./output_images/video.jpg
[heat]: ./output_images/heat.jpg

### Extracting features
####1. Combining Spatial binning, Color and HOG features 

The code for this step is contained in the second and third code cell of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][car_not_car]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][Car_HOG]
![alt text][Car-CH-1-Spatial-binning]
![alt text][Car-CH-1-Color-histogram]
![alt text][not-Car_CH-1_and_HOG]
![alt text][not-Car_CH-1_Spatial_inning]
![alt text][not-Car_CH-1_Color_histogram]

####2. Choose feature parameters.

As you can see in the example above, even going all the way down to 32*32 pixel resolution, the car itself is still clearly identifiable by eye, and this means that the relevant features are still preserved at this resolution.So I choose to convert images with size (32,32) to one dimensional feature vector.

I add color histograms of color to the training feature. And the number of bins I choosed is 32. The color is a feature of the vehicle. Usually, background color is more chaotic,and the color of vehicle is more uniform. This is the reason that it can be used as a feature.  

I tried various combinations of parameters to training SVM, and choose the group that has the highest accuracy. The training set I used is [vehicle_smallset](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles_smallset.zip) and  [not-vehicles_smallset](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles_smallset.zip), which contained 1196 and 1125 pictures respectively.

Below is the code fragment I tried to training SVM:

		for orient in range(4,13,1):
        	for pix_per_cell in range(3,12,1):
          		for cell_per_block in range(2,5,1):

The highest accuracies are listed below:

| orientations | pixels_per_cell| cells_per_block | Feature vector length | accuracy |
| ------------ | -------------- | --------------- | --------------------- | -------- |
| 2 | 10 | 4 | 4032 | 0.9957 |
| 5 | 10 | 3 | 5328 | 0.9914 |
| 5 | 11 | 4 | 4128 | 0.9914 |
| 6 | 11 | 2 | 4320 | 0.9914 |
| 7 | 10 | 4 | 6192 | 0.9957 |
| 8 | 8  | 2 | 7872 | 0.9914 |
| 8 | 11 | 2 | 4704 | 0.9935 |
| 9 | 9  | 2 | 7056 | 0.9957 |

Finally, I use training set [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and  [not-vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) to test their accuracy. And the highest accuracy is 0.9758, and the parameter is (orient=8, pixels_per_cell=8,cells_per_block=2). I select these parameters to training SVM. 

####3. Training a classifier using selected features combination

The code for this step is contained in the forth code cell of the IPython notebook. 

I choose the combination of spatial binning, histograms of color and HOG feature as my training feature, which is a vector of lenth 7872. The training set I used is [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and  [not-vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip), which contained 8972 and 8968 pictures respectively. As you can see, it is a balanced dataset, which can avoid the algorithm simply classify everything as belonging to the majority class. 

First, I use the functions I defined, namely, bin_spatial(), color_hist(), get_hog_feature, and extract_features(). I then read in our car and non-car images, extract the features for each. I normalize the features with the StandardScaler() method which provided by sklearn package. And then, I use Scikit-Learn train_test_split() function to shuffle and split the data into training and testing sets with rate of 0.7. Finally, I use the training data to train a Linear Support Vector Machine and test it's accuracy. And I got an accuracy of 0.9758.


###Sliding Window Search

####1. implementing a sliding window search

The picture captured by the camera in front of the car can be divided into two parts. In the upper section, it is usually the sky. And in the lower, it is road pavement which we need to detect. So, I set y_start_stop as range of [400, 600].  Obviously, the farther away, see the smaller things. So, in the upper detect area, I take small search window size. In the lower detect area, I take big search window size. The range of search window size is from (64,64) to (128, 128). The xy_overlap is (0.6,0.6). And the number of windows is 153. 

| y_start_stop |  window size |
| ------------ | ------------ |
| [400,500] | (64,64) |
| [400,500] | (96,96) |
| [450,600] | (128, 128) |


![alt text][slide_window]

####2. some examples

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector.I recorded the positions of positive detections in each test image.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here are some example images:

![alt text][example]
![alt text][heat]

In video proprecessing, I recorded the positions of positive detections in 6 previous frames, and the thresholded that map to identify vehicle positions in the video.

---

### Video Implementation

####1. Final video output.
![alt text][video_img]
Here's a [link to my video result](./project_output.mp4)

---

###Discussion

I think the robustness of my pipeline need to be improved.First, the SVM accuracy is just 0.9758. It is too low, and can't classify cars in real road well. Second, the way of detecting the car area is too simple. I think we can improve it by applying different filters,morphological operations, contour algorithms, and validations.
 

