import os
import numpy as np
import pandas as pd
import random as rng
import cv2 as cv


# Obtain the path 
dataset_path = "/Volumes/My-Passport/Dataset/CUB_200_2011/CUB_200_2011/"
mask_rcnn = "/Users/saravana/Documents/Work/Projects/Bird_Detection/mask-rcnn"
midas_large_path = "/Users/saravana/Documents/Work/Projects/Bird_Detection/midas/model-f6b98070.onnx"
midas_small_path = "/Users/saravana/Documents/Work/Projects/Bird_Detection/midas/model-small.onnx"


test_train_data = os.path.sep.join([dataset_path,"train_test_split.txt"])
image_data = os.path.sep.join([dataset_path,"images.txt"])

# load the COCO class labels our Mask R-CNN was trained on
labelsPath = os.path.sep.join([mask_rcnn,"mscoco_labels.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

# derive the paths to the Mask R-CNN weights and model configuration
weightsPath = os.path.sep.join([mask_rcnn,"frozen_inference_graph.pb"])
configPath = os.path.sep.join([mask_rcnn,"mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])


#Minimum Confidence score of 95% for bird detection
# confidence_score = 0.95
confidence_score = 0.50
#Minimum threshold for pixel-wise mask segmentation
pixel_seg_threshold = 0.3
# Number of GrabCut iterations (larger value => slower runtime)
no_of_grabcut_iter = 10


# load our Mask R-CNN trained on the COCO dataset (90 classes)
# from disk
def load_model():
    print("[INFO] loading Mask R-CNN from disk...")
    model = cv.dnn.readNetFromTensorflow(weightsPath, configPath)
    return model

def load_from(csv):
    # Sample of 3000 labelled images
    # Number of flying birds = 302
    # Number of non-flying birds = 2698
    sample_images_df = pd.read_csv(csv, 
                            header=None, 
                            delimiter=',',
                            names=['image_id', 'image_name', 'is_training', 'sift_keypoints', 'truth_is_flying', 'pred_is_flying'])
    # Dropping header(first row) from the data frame
    sample_images_df = sample_images_df.iloc[1: , :]
    return sample_images_df

# The purpose of non-max suppression is to select the best bounding box for an object and 
# reject or “suppress” all other bounding boxes. 
# The NMS takes two things into account
# The objectiveness score is given by the model
# The overlap or IOU of the bounding boxes
# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,3]
	y1 = boxes[:,4]
	x2 = boxes[:,5]
	y2 = boxes[:,6]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("float")    



def load_3k_images_from(csv):
    # Sample of 3000 labelled images
    # Number of flying birds = 302
    # Number of non-flying birds = 2698
    sample_images_df = pd.read_csv(csv, 
                            header=None, 
                            delimiter=',',
                            names=['image_id', 'image_name', 'is_training', 'sift_keypoints', 'truth_is_flying'])

    # Dropping header(first row) from the data frame
    sample_images_df = sample_images_df.iloc[1: , :]
    return sample_images_df

def load_data_for_ratio_method():
    df = load_3k_images_from('ratio_method_images.csv')
    #Adding new column 'pred_is_flying' with 'nan' to the data frame 
    # df['pred_is_flying'] = np.nan
    # df['pred_probas'] = np.nan
    # df['pred_probas'] = np.empty((len(df), 0)).tolist()
    df['ratio'] = np.nan
    return df

# def annotate_largest_contour(mask,image,threshold,depth):
def annotate_largest_contour(mask,image,boxW,boxH):
    mask_ratio = None

    imgWidth = image.shape[1]

    bb_box_to_img_size_ratio = ((boxW+boxH)/2)/imgWidth

    # Contour detection method
    rng.seed(12345)
    ret, thresh = cv.threshold(mask, 127, 255, 0)

    # Detect edges using Canny
    canny_output = cv.Canny(image, 100, 200)

    #Find contours for detected edges
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    ratios = []
    areas = []
    circumferences = []

    # Find area and circumference for all the detected contours. Compute ratio of Area/circumference. 
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        area = cv.contourArea(contours[i])
        circumference = cv.arcLength(contours[i],True)
        if area > 0:
            ratio_area_to_circumference = area/circumference
            # if ratio_area_to_circumference >= 5:
            ratios.append(ratio_area_to_circumference)
            areas.append(area)
            circumferences.append(circumference)

    # Select the contour with highest ratio and set values 0/1 to 'is_flying' variable
    if len(ratios) > 0:
        ratios_numpy = np.array(ratios)
        areas_numpy = np.array(areas)
        circumferences_numpy = np.array(circumferences) 

        ratio = np.amax(ratios_numpy)
        area = np.amax(areas_numpy)
        circumference = np.amax(circumferences_numpy)
        
        print("Ratio before depth: ",ratio)
        # if depth > 1.0:
        #     difference = depth - 1.0
        #     ratio = (ratio * difference) + ratio
        # else:
        #     ratio = ratio * depth
        
        # mask_ratio = ratio
        # mask_ratio = ratio * depth

        ratio = bb_box_to_img_size_ratio * ratio
        mask_ratio = ratio
    else:
        print("No contours found on image-{}",image)

    return mask_ratio
