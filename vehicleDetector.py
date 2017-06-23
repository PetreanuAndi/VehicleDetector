import os
import glob
import numpy as np
import cv2
import time
import pickle
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from readDataset import getDatabaseStruct
from hogFeatureExtractor import extract_features
from hogFeatureExtractor import draw_boxes
from hogFeatureExtractor import slide_window
from hogFeatureExtractor import search_windows
from hogFeatureExtractor import get_hog_features
from hogFeatureExtractor import bin_spatial
from hogFeatureExtractor import color_hist
from classifier import train_classifier
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip


PATH_TO_EXAMPLES = 'test_images/*'
PATH_TO_OUTPUT_BOXES = './output_boxes/'

global X_scaler
global svc
global prev_labels

def convert_color(img, conv='RGB2YCrCb'):
	if conv == 'RGB2YCrCb':
		return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
	if conv == 'BGR2YCrCb':
		return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
	if conv == 'RGB2LUV':
		return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


def extractEntireHogMap(img,scale=1):

	# use the global, pre-computed classifier and scaler
	global X_scaler
	global svc

	# parameters for search and features
	color_space = 'YCrCb'
	orient=9
	pix_per_cell=8
	cell_per_block=2
	hog_channel='ALL'
	spatial_size=(64,64)
	hist_bins=64
	spatial_feat=True
	hist_feat=True
	hog_feat=True
	window=64

	# determine masked area for optimal search
	ystart = 400
	ystop = 656
	xstart = 600
	xstop = 1240
	
	img_tosearch = img[ystart:ystop,xstart:,:]
	cv2.imwrite('test.jpg',img_tosearch)

	img_boxes = []
	t=time.time()
	count=0

	draw_img = np.copy(img)
	heatmap = np.zeros_like(img[:,:,0])
	
	ctrans_tosearch = convert_color(img_tosearch,conv='RGB2YCrCb')
	# apply scale to entire window <=> searching with different window sizes
	if (scale!=1):
		imshape = ctrans_tosearch.shape
		ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

	ch1 = ctrans_tosearch[:,:,0]
	ch2 = ctrans_tosearch[:,:,1]
	ch3 = ctrans_tosearch[:,:,2]

	# number of hog cells across the image
	nxblocks = (ch1.shape[1] // pix_per_cell)-1
	nyblocks = (ch1.shape[0] // pix_per_cell)-1

	nfeat_per_block = orient*cell_per_block**2

	nblocks_per_window = (window // pix_per_cell)-1
	cells_per_step = 2 # Instead of overlap

	nxsteps = (nxblocks - nblocks_per_window)//cells_per_step
	nysteps = (nyblocks - nblocks_per_window)//cells_per_step

	# extract hog features per channel
	hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
	hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
	hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

	pred_idx=0

	for xb in range(nxsteps):
		for yb in range(nysteps):
			count +=1
			ypos = yb*cells_per_step
			xpos = xb*cells_per_step

			# extract patches of hog features, per channel
			hog_feat1 = hog1[ypos:ypos+nblocks_per_window , xpos:xpos+nblocks_per_window].ravel()
			hog_feat2 = hog2[ypos:ypos+nblocks_per_window , xpos:xpos+nblocks_per_window].ravel()
			hog_feat3 = hog3[ypos:ypos+nblocks_per_window , xpos:xpos+nblocks_per_window].ravel()

			# compute concatenated hog_features
			hog_features = np.concatenate((hog_feat1,hog_feat2,hog_feat3),axis=0)

			xleft = xpos*pix_per_cell
			ytop = ypos*pix_per_cell

			# extract image patch
			subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

			# extract spatial/histogram features for image patch
			spatial_features = bin_spatial(subimg,size=spatial_size)
			hist_features = color_hist(subimg,nbins=hist_bins)

			# create entire feature vector
			global_features = []
			global_features.append(spatial_features)
			global_features.append(hist_features)
			global_features.append(hog_features)

			test_features = X_scaler.transform(np.array(np.concatenate(global_features)).reshape(1,-1))
			test_prediction = svc.predict(test_features)

			if(test_prediction==1):
				pred_idx+=1
				xbox_left = np.int(xleft*scale)
				ytop_draw = np.int(ytop*scale)
				win_draw = np.int(window*scale)

				#img_name = 'img_'+str(pred_idx)+'.jpg'
				#heat_name = 'heat_'+str(pred_idx)+'.jpg'

				cv2.rectangle(draw_img,(xbox_left+xstart,ytop_draw+ystart),(xbox_left+xstart+win_draw,ytop_draw+win_draw+ystart),(0,0,255))
				#cv2.imwrite(PATH_TO_OUTPUT_BOXES+img_name,draw_img)
				#print('Wrote to file : ',PATH_TO_OUTPUT_BOXES+img_name)

				img_boxes.append(((xbox_left+xstart,ytop_draw+ystart),(xbox_left+xstart+win_draw,ytop_draw+win_draw+ystart)))
				heatmap[ytop_draw+ystart:ytop_draw+win_draw+ystart , xbox_left+xstart:xbox_left+xstart+win_draw] +=1

				#cv2.imwrite(PATH_TO_OUTPUT_BOXES+heat_name,heatmap*10)

	return draw_img,heatmap

def apply_threshold(heatmap,threshold=4):
	heatmap[heatmap<=threshold] = 0
	return heatmap

def draw_labeled_boxes(img,labels):

	for car_number in range(1,labels[1]+1):
		nonzero = (labels[0]==car_number).nonzero()

		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])

		bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox),np.max(nonzeroy)))

		cv2.rectangle(img,bbox[0],bbox[1],(0,255,255),3)

	return img

# process each frame in video
def process_image(img):

	global prev_labels

	#img = cv2.imread(img_src)
	image_original = np.copy(img)
	#img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

	# get entire hog maps for multiple scales, then combine
	out_img1 , heat_map1 = extractEntireHogMap(img,scale=1.5)
	out_img2 , heat_map2 = extractEntireHogMap(img,scale=1.0)
	out_img3 , heat_map3 = extractEntireHogMap(img,scale=2.0)
	
	# combine scales and threshold
	heat_map = heat_map1+heat_map2+heat_map3
	heat_map = apply_threshold(heat_map,threshold=2)

	# extract labels 
	labels = label(heat_map)

	# in order to stop the flickering, keep a short history of labels and fill in.
	if (labels[1]==0 and prev_labels!=None):
		labels=prev_labels

	# draw heat boxes
	draw_img = draw_labeled_boxes(image_original,labels)
	prev_labels=labels
	#cv2.imwrite('img.jpg',draw_img)
	#cv2.imwrite('img_out.jpg',out_img3)
	return draw_img

def main():

	global X_scaler
	global svc
	global prev_labels
	prev_labels=None

	# if we have already trained the classifier just load from disk
	if not os.path.exists('./classifier.pkl'):

		print('-'*50)
		print('Training classifier now')

		# get vehicle/non-vehicle examples for SVM
		cars,notcars = getDatabaseStruct()

		# classifier parameters
		color_space = 'YCrCb'
		orient=9
		pix_per_cell=8
		cell_per_block=2
		hog_channel='ALL'
		spatial_size=(32,32)
		hist_bins=32
		spatial_feat=True
		hist_feat=True
		hog_feat=True

		# --------------------------------------------------------------------------------------------------
		# train classifier 

		t = time.time()
		n_samples = len(cars)
		random_idxs = np.random.randint(0,len(cars), n_samples)

		test_cars = cars#np.array(cars)[random_idxs]
		test_notcars = notcars#np.array(notcars)[random_idxs]

		car_features = extract_features(test_cars,color_space,spatial_size,hist_bins,
									orient,pix_per_cell,cell_per_block,hog_channel,spatial_feat,
									hist_feat,hog_feat)

		notcar_features = extract_features(test_notcars,color_space,spatial_size,hist_bins,
										orient,pix_per_cell,cell_per_block,hog_channel,spatial_feat,
										hist_feat,hog_feat)

		print(time.time()-t, 'Seconds to compute features ...')

		svc , X_scaler = train_classifier(car_features,notcar_features,
									orient,pix_per_cell,cell_per_block,hist_bins,spatial_size)

		with open('classifier.pkl','wb') as f:
			pickle.dump(svc,f)
		with open('scaler.pkl','wb') as f:
			pickle.dump(X_scaler,f)

	else:
		with open('classifier.pkl','rb') as f:
			svc = pickle.load(f)
		with open('scaler.pkl','rb') as f:
			X_scaler = pickle.load(f)

	# ---------------------------------------------------------------------------------------------------
	# Per image hog

	# extractEntireHogMap(example_images[1],window=64)
	# process_image(example_images[3])


	print('-'*50)
	print('Processing project video')

	output = 'output2.mp4'
	clip = VideoFileClip("project_video.mp4")
	test_clip = clip.fl_image(process_image)
	test_clip.write_videofile(output,audio=False)

	# --------------------------------------------------------------------------------------------------
	# Per patch Hog 
	#example_images = glob.glob(PATH_TO_EXAMPLES)

	#images = []
	#titles = []
	# restrict area 
	#y_start_stop = [400,656]
	#overlap=0.5
	
	#for img_src in example_images:
	#	t1=time.time()
	#	img_name = img_src.strip().split('/')[-1]
	#	print(img_name)
	#	img = cv2.imread(img_src)
	#	draw_img = np.copy(img)

		#img = img.astype(np.float32)/255

	#	print(np.min(img), np.max(img))

	#	windows = slide_window(img, x_start_stop=[None,None], y_start_stop=y_start_stop,
	#							xy_window=(96,96),xy_overlap=(overlap,overlap))

	#	hot_windows = search_windows(img, windows, svc, X_scaler, color_space,
	#								spatial_size,hist_bins,hist_range = (0,256),orient=orient,
	#								pix_per_cell=pix_per_cell,cell_per_block=cell_per_block,
	#								hog_channel=hog_channel,spatial_feat=spatial_feat,
	#								hist_feat=hist_feat,hog_feat=hog_feat)

	#	window_img = draw_boxes(img_name, draw_img, hot_windows, color=(0,0,255), thick=5)
	#	images.append(window_img)
	#	titles.append('')
	#	print(time.time()-t1,' seconds to process one image searching ',len(windows),' windows')

	

if __name__ == '__main__':
    main()