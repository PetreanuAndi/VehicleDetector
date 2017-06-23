
**Vehicle Detection Project**

[//]: # (Image References)
[image1]: ./output_images/Car_features.png
[image2]: ./output_images/NotCar_features.png
[image3]: ./output_images/img_7.jpg
[image4]: ./output_images/heat_6.jpg
[video1]: ./output.mp4
 

---

###Histogram of Oriented Gradients (HOG)

Methods for computing HOG can be gound both in "hogFeatureExtractor.py" and in "vehicleDetector.py"
The ladder serves as the main functionality file in the project.
"readDataset.py" is used to extract the images from the car/noncar dataset with glob functionality.

![alt text][image1]
![alt text][image2]

Initially, I computed hog for sub-samples of images, taken from the sliding window search.
The lectures provided helped me optimise this process by computing HOG on the entire image and then subsampling the HOG mask. (along with hist and spatial features). This part can be found in "extractEntireHogMap(image,scale)" method, in vehicleDetector.py



I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


####2. Explain how you settled on your final choice of HOG parameters.

Starting with the suggested parameters, i tried different combinations of hog_channels, spatial_size, orientation and color space for both training the classifier and for predicting the output of the SVM. 

While a variation of this parameter space gave me the best SVM accuracy (0.997), i had to experiment further in order to get it to work best at the resolution that i was using when predicting with sliding window. 

My final parameter space is : color_space = 'YCrCb',orient=9,pix_per_cell=8,cell_per_block=2,hog_channel='ALL',spatial_size=(64,64),hist_bins=64, spatial_feat=True,hist_feat=True,hog_feat=True,window=64

This seemes to work best with a heat_map threshold of 2 and at 3 different scales : 1.0 / 1.5 / 2.0 over a window of 64.


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using 3 feature vectors (spatial, hist and hog) in "vehicleDetector.py" and as a test in "classifier.py"
While the training and testing times were pretty large, i saved the model (classifier/scaler) in pickles and would only recompute when i did not find those files in root folder.

I had an interesting issue while reading with cv2.imread. I did not realise at first that i was training on BGR and then testing on RGB (converting all to YCrCb) and it did not perform very well. This issue killed a lot of my time on the project, and i realised it by mistake basically.

###Sliding Window Search

Sliding window search is implemented in "hogFeatureExtractor.py" under a method with the same name.
I only performed the search in the lower half /right half of the image, because that was the relevant space where cars appeared and i had less windows to go through => better solution iterations.

While the window size remains fixed at 64, i used different scales for the images (trained at different scales + extractEntireHogMap() on 3 different scales, then combined their heatmap, then thresholded)

![alt text][image3]

### Video Implementation

I used the callback functionality of moviepy.editor - VideoFileClip library and performed all the pipeline steps inside of "process_image(img)"
Here's a [link to my video result](./output.mp4)

False positives were removed as a result of the heat_map thresholding. I experimented with various thresholding parameters, but found that larger values would segment my output quite badly. I used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap

---

###Discussion

I think that the feature extractor of (hog/hist and spatial features) can be easily replaced with a CNN like VGG and then on top of that one can apply a SVM to differentiate between positives and negatives. This way, the CNN will learn its own filters and features, probably much more complex and less time consuming than HOG

