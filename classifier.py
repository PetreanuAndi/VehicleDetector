import numpy as np
import cv2
import time
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
from readDataset import getDatabaseStruct
from hogFeatureExtractor import extract_features



def train_classifier(car_features,notcar_features , orient,pix_per_cell,cell_per_block,hist_bins,spatial_size):

	X = np.vstack((car_features,notcar_features)).astype(np.float64)
	X_scaler = StandardScaler().fit(X)
	scaled_X = X_scaler.transform(X)

	y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

	rand_state = np.random.randint(0,100)
	X_train,X_test,y_train,y_test = train_test_split(scaled_X,y,test_size=0.1,random_state=rand_state)

	print('Using: ',orient,' orientations, ',pix_per_cell,' pixels per cell, ',
			cell_per_block,' cells per block, ',hist_bins,' histogram_bins, ', spatial_size,
			'spatial sampling')
	print('Feature Vector Length:',len(X_train[0]))

	svc = LinearSVC()

	t=time.time()
	svc.fit(X_train,y_train)
	print(round(time.time()-t,2),' Seconds to train SVC...')

	print('Test Accuracy of SVC = ', round(svc.score(X_test,y_test),4))

	return svc, X_scaler

def main():
	cars,notcars = getDatabaseStruct()

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
	hog_channel='ALL'
	spatial_size=(16,16)
	hist_bins=16
	spatial_feat=True
	hist_feat=True
	hog_feat=True


	t = time.time()
	n_samples = len(cars)
	random_idxs = np.random.randint(0,len(cars), n_samples)

	test_cars = np.array(cars)[random_idxs]
	test_notcars = np.array(notcars)[random_idxs]

	car_features = extract_features(test_cars,color_space,spatial_size,hist_bins,
					orient,pix_per_cell,cell_per_block,hog_channel,spatial_feat,
					hist_feat,hog_feat)

	notcar_features = extract_features(test_notcars,color_space,spatial_size,hist_bins,
										orient,pix_per_cell,cell_per_block,hog_channel,spatial_feat,
										hist_feat,hog_feat)

	print(time.time()-t, 'Seconds to compute features ...')

	train_classifier(car_features,notcar_features,orient,pix_per_cell,cell_per_block,hist_bins,spatial_size)


if __name__ == '__main__':
    main()