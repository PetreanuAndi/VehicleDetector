import numpy as np
import cv2
import time
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
from readDataset import getDatabaseStruct


PATH_TO_OUTPUT_BOXES = './output_boxes/'

def get_hog_features(img,orient,pix_per_cell,cell_per_block,vis=False,feature_vec=True):
	
	if vis == True:
		features,hog_image = hog(img,orientations=orient,pixels_per_cell=(pix_per_cell,pix_per_cell),
									cells_per_block = (cell_per_block,cell_per_block),
									transform_sqrt=False,
									visualise=vis,feature_vector=feature_vec)
		return features,hog_image

	features = hog(img,orientations=orient,pixels_per_cell=(pix_per_cell,pix_per_cell),
					cells_per_block = (cell_per_block,cell_per_block),
					transform_sqrt=False,
					visualise=vis,feature_vector=feature_vec)
	return features

#downsample
def bin_spatial(img,size=(32,32)):
	color1 = cv2.resize(img[:,:,0],size).ravel()
	color2 = cv2.resize(img[:,:,1],size).ravel()
	color3 = cv2.resize(img[:,:,2],size).ravel()
	return np.hstack((color1,color2,color3))


def color_hist(img,nbins=32):
	channel1_hist = np.histogram(img[:,:,0],bins=nbins)
	channel2_hist = np.histogram(img[:,:,1],bins=nbins)
	channel3_hist = np.histogram(img[:,:,2],bins=nbins)
	hist_features = np.concatenate((channel1_hist[0],channel2_hist[0],channel3_hist[0]))

	return hist_features

def extract_features(imgs, color_space='RGB', spatial_size=(32,32),hist_bins=32,orient=9,
						pix_per_cell=8,cell_per_block=2,hog_channel=0, spatial_feat=True,hist_feat=True,hog_feat=True):

	features = []

	for file in imgs:
		file_features = []
		image = cv2.imread(file)
		image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

		if color_space != 'RGB':
			if color_space =='HSV':
				feature_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
			elif color_space =='LUV':
				feature_image = cv2.cvtColor(image,cv2.COLOR_RGB2LUV)
			elif color_space =='HLS':
				feature_image = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
			if color_space =='YUV':
				feature_image = cv2.cvtColor(image,cv2.COLOR_RGB2YUV)
			if color_space =='YCrCb':
				feature_image = cv2.cvtColor(image,cv2.COLOR_RGB2YCrCb)
		else: feature_image = np.copy(image)

		if spatial_feat == True:
			spatial_features = bin_spatial(feature_image,size=spatial_size)
			file_features.append(spatial_features)

		if hist_feat == True:
			hist_features = color_hist(feature_image,nbins=hist_bins)
			file_features.append(hist_features)

		if hog_feat == True:
			if hog_channel == 'ALL':
				hog_features = []
				for channel in range(feature_image.shape[2]):
					hog_features.append(get_hog_features(feature_image[:,:,channel],
										orient,pix_per_cell,cell_per_block,vis=False,feature_vec=True))
				hog_features = np.ravel(hog_features)
			else:
				hog_features = get_hog_features(feature_image[:,:,hog_channel],orient,
										pix_per_cell,cell_per_block,vis=False,feature_vec=True)
			file_features.append(hog_features)
		features.append(np.concatenate(file_features))

	return features

def slide_window(img, x_start_stop=[None,None], y_start_stop=[None,None],
					xy_window=(64,64),xy_overlap=(0.5,0.5)):
	
	if x_start_stop[0]==None:
		x_start_stop[0]=0
	if x_start_stop[1]==None:
		x_start_stop[1]=img.shape[1]
	if y_start_stop[0]==None:
		y_start_stop[0]=0
	if y_start_stop[1]==None:
		y_start_stop[1]=img.shape[0]

	xspan = x_start_stop[1] - x_start_stop[0]
	yspan = y_start_stop[1] - y_start_stop[0]

	nx_pix_per_step = np.int(xy_window[0]*(1-xy_overlap[0]))
	ny_pix_per_step = np.int(xy_window[1]*(1-xy_overlap[1]))

	nx_windows = np.int(xspan/nx_pix_per_step)-1
	ny_windows = np.int(yspan/ny_pix_per_step)-1

	window_list=[]

	for ys in range(ny_windows):
		for xs in range(nx_windows):
			startx = xs*nx_pix_per_step + x_start_stop[0]
			endx = startx + xy_window[0]

			starty = ys*ny_pix_per_step + y_start_stop[0]
			endy = starty + xy_window[1]

			window_list.append(((startx,starty),(endx,endy)))	
	
	return window_list

def draw_boxes(img_name, img, bboxes, color=(0,0,255), thick=5):
	imcopy = np.copy(img)

	for bbox in bboxes:
		cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
		cv2.imwrite(PATH_TO_OUTPUT_BOXES + img_name,imcopy)
		print('Draw boxes to ',PATH_TO_OUTPUT_BOXES + img_name)

	return imcopy

def single_img_features(img, color_space='RGB', spatial_size=(32,32),
						hist_bins = 32, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
						spatial_feat=True,hist_feat=True,hog_feat=True,vis=False):

	img_features = []

	if color_space != 'RGB':
		if color_space =='HSV':
			feature_image = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
		elif color_space =='LUV':
			feature_image = cv2.cvtColor(img,cv2.COLOR_RGB2LUV)
		elif color_space =='HLS':
			feature_image = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
		if color_space =='YUV':
			feature_image = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
		if color_space =='YCrCb':
			feature_image = cv2.cvtColor(img,cv2.COLOR_RGB2YCrCb)
	else: feature_image = np.copy(img)


	if spatial_feat == True:
		spatial_features = bin_spatial(feature_image,size=spatial_size)
		print('Spatial features : ',spatial_features.shape)
		img_features.append(spatial_features)

	if hist_feat == True:
		hist_features = color_hist(feature_image,nbins=hist_bins)
		print('Hist_Features : ',hist_features.shape)
		img_features.append(hist_features)

	if hog_feat == True:
		if hog_channel == 'ALL':
			hog_features = []
			for channel in range(feature_image.shape[2]):
				hog_features.append(get_hog_features(feature_image[:,:,channel],
									orient,pix_per_cell,cell_per_block,vis=False,feature_vec=True))
			hog_features = np.concatenate(hog_features)
		else:
			if vis == True:
				hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel],orient,
														pix_per_cell,cell_per_block,vis=True,feature_vec=True)
			else:
				hog_features = get_hog_features(feature_image[:,:,hog_channel],orient,
									pix_per_cell,cell_per_block,vis=False,feature_vec=True)
		print('Hog features : ',hog_features.shape)
		img_features.append(hog_features)
	
	if vis == True:
		return np.concatenate(img_features), hog_image
	else:
		print('IMG features : ',np.concatenate(img_features).shape)
		return np.concatenate(img_features)

def search_windows(img, windows, clf, scaler, color_space='RGB',
					spatial_size=(32,32), hist_bins=32, hist_range = (0,256), orient=9,
					pix_per_cell=8, cell_per_block=2,hog_channel=0, spatial_feat=True,
					hist_feat=True, hog_feat=True):

	on_windows = []
	for window in windows:

		test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64,64))

		features = single_img_features(test_img, color_space=color_space,spatial_size=spatial_size,
										hist_bins=hist_bins,orient=orient,pix_per_cell=pix_per_cell,
										cell_per_block=cell_per_block,hog_channel=hog_channel,spatial_feat=spatial_feat,
										hist_feat = hist_feat,hog_feat=hog_feat,vis=False)

		print('Features shape : ')
		print(features.shape)

		test_features = scaler.transform(np.array(features).reshape(1,-1))

		prediction = clf.predict(test_features)

		if prediction==1:
			on_windows.append(window)

	return on_windows

def main():


	cars,notcars = getDatabaseStruct()
	print(len(cars))
	car_ind = np.random.randint(0,len(cars))
	notcar_ind = np.random.randint(0,len(notcars))

	car_image = cv2.imread(cars[car_ind])
	#car_image = cv2.cvtColor(car_image,cv2.COLOR_BGR2RGB)

	notcar_image = cv2.imread(notcars[notcar_ind])
	#notcar_image = cv2.cvtColor(notcar_image,cv2.COLOR_BGR2RGB)

	color_space = 'YCrCb'
	orient=9
	pix_per_cell=8
	cell_per_block=2
	hog_channel=0
	spatial_size=(16,16)
	hist_bins=16
	spatial_feat=True
	hist_feat=True
	hog_feat=True

	car_features, car_hog_image = single_img_features(car_image,color_space,spatial_size,hist_bins,
									orient,pix_per_cell,cell_per_block,hog_channel,spatial_feat,
									hist_feat,hog_feat,vis=True)

	notcar_features, notcar_hog_image = single_img_features(notcar_image,color_space,spatial_size,hist_bins,
										orient,pix_per_cell,cell_per_block,hog_channel,spatial_feat,
										hist_feat,hog_feat,vis=True)


	cv2.imshow('Car',car_image)

	car_hog_image *= 10.0
	notcar_hog_image *= 10.0

	car_hog_image = np.dstack((car_hog_image, car_hog_image, car_hog_image))
	notcar_hog_image = np.dstack((notcar_hog_image, notcar_hog_image, notcar_hog_image))

	print(car_image.shape)
	print(car_hog_image.shape)

	output = np.hstack((car_image,car_hog_image))
	output = cv2.resize(output,(0,0), fx=2.0,fy=2.0)

	cv2.imshow('Car_features',output/255.)
	cv2.imwrite('Car_features.png', output)

	output2 = np.hstack((notcar_image, notcar_hog_image))
	output2 = cv2.resize(output2,(0,0), fx=2.0,fy=2.0)

	cv2.imshow('NotCar_features',output2/255.)
	cv2.imwrite('NotCar_features.png', output2)

	cv2.waitKey(0)
	cv2.destroyAllWindows()





if __name__ == '__main__':
    main()












