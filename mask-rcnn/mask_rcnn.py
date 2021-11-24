#Model: Mask-RCNN
#Implementation of Background subtraction by combining Mask-rcnn and Grabcut algorithm

# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2 as cv
import os
import csv


# Write the values to given filename     
def write_to_csv(filename,header=None,data=None):
    # Open file in append mode
    with open(filename, 'a+', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        if header != None:
            writer.writerow(header)
        # write multiple rows
        if data != None:
            writer.writerows(data)

# Filter only bounding boxes of birds class objects only
def filter_bounding_boxes(boxes):
    # Birds-class_id = 15 
    classID = 15
    each_class_id = int(boxes[1])
    if (each_class_id == classID):
        return True
    else:
        return False   


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
    
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mask-rcnn", required=True,
                help="base path to mask-rcnn directory")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
# ap.add_argument("-c", "--confidence", type=float, default=0.90,
#                 help="minimum probability to filter weak detections")
ap.add_argument("-c", "--confidence", type=float, default=0.50,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="minimum threshold for pixel-wise mask segmentation")
ap.add_argument("-u", "--use-gpu", type=bool, default=0,
                help="boolean indicating if CUDA GPU should be used")
ap.add_argument("-e", "--iter", type=int, default=10,
                help="# of GrabCut iterations (larger value => slower runtime)")
args = vars(ap.parse_args())


# load the COCO class labels our Mask R-CNN was trained on
#labelsPath = os.path.sep.join([args["mask_rcnn"],
#                               "object_detection_classes_coco.txt"])
labelsPath = os.path.sep.join([args["mask_rcnn"],
                               "mscoco_labels.names"])
LABELS = open(labelsPath).read().strip().split("\n")



# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

# derive the paths to the Mask R-CNN weights and model configuration
weightsPath = os.path.sep.join([args["mask_rcnn"],
                                "frozen_inference_graph.pb"])
configPath = os.path.sep.join([args["mask_rcnn"],
                               "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

# load our Mask R-CNN trained on the COCO dataset (90 classes)
# from disk
print("[INFO] loading Mask R-CNN from disk...")
model = cv.dnn.readNetFromTensorflow(weightsPath, configPath)    
    
# load our input image from disk and display it to our screen
image = cv.imread(args["image"])
# image = imutils.resize(image, width=600)
#Resize the image, Some images are different sizes. (Resizing is very Important)
print("Shape is {}".format(image.shape))
aspect_ratio = image.shape[0]/image.shape[1]
print("Aspect ratio is {}".format(aspect_ratio))
# image = cv.resize(image, (1000, 1000), interpolation = cv.INTER_AREA)
# image = cv.resize(image, (400, 400), interpolation = cv.INTER_AREA)

# Scaling up the image 1.511 times specifying a single scale factor.
scale_up = 1.53
image = cv.resize(image, None, fx= scale_up, fy= scale_up, interpolation= cv.INTER_LINEAR)


cv.imshow("Input", image)
cv.waitKey(0)
 

header = ['imagepath', 'is_flying', 'ratio', 'area', 'circumference']
write_to_csv('bird_ratio.csv',header,None)


# construct a blob from the input image and then perform a
# forward pass of the Mask R-CNN, giving us (1) the bounding box
# coordinates of the objects in the image along with (2) the
# pixel-wise segmentation for each specific object
blob = cv.dnn.blobFromImage(image, swapRB=True, crop=False)
model.setInput(blob)
(boxes, masks) = model.forward(["detection_out_final","detection_masks"])


# Extract only the bounding boxes values only for 'bird' class
# Convert the obtained f_boxes filter object of np.array
f_boxes = filter(filter_bounding_boxes, boxes[0,0])
filtered_boxes = np.array(list(f_boxes))
print("Filtered Box")
print(len(filtered_boxes))
print("------------")
# convert filtered bounding boxes of type np.array to list
detected_boxes = boxes[0,0].tolist()

print(filtered_boxes)
filtered_boxes = non_max_suppression_fast(filtered_boxes,0.50)


# loop over the number of detected (only bird class) objects
for i in range(0,filtered_boxes.shape[0]):
    # convert filtered(only bird class) bounding boxes of type np.array to list
    filtered = filtered_boxes[i].tolist()
    # find the index of bird class from the detected bounding boxes of all classes 
    index = detected_boxes.index(filtered)
    
    # extract the class ID of the detection along with the
    # confidence (i.e., probability) associated with the
    # prediction
    classID = int(filtered_boxes[i, 1])
    confidence = filtered_boxes[i, 2]
    print("Confidence {}".format(confidence))
    # filter out weak predictions by ensuring the detected
    # probability is greater than the minimum probability
#     if confidence > args["confidence"]:
    # show the class label
    print("[INFO] showing output for '{}'...".format(LABELS[classID]))
    # scale the bounding box coordinates back relative to the
    # size of the image and then compute the width and the
    # height of the bounding box
    (H, W) = image.shape[:2]
    box = filtered_boxes[i, 3:7] * np.array([W, H, W, H])
    (startX, startY, endX, endY) = box.astype("int")
    boxW = endX - startX
    boxH = endY - startY

    # extract the pixel-wise segmentation for the object, resize
    # the mask such that it's the same dimensions as the bounding
    # box, and then finally threshold to create a *binary* mask
    mask = masks[index, classID]
    mask = cv.resize(mask, (boxW, boxH),interpolation=cv.INTER_CUBIC)
    mask = (mask > args["threshold"]).astype("uint8") * 255
    # allocate a memory for our output Mask R-CNN mask and store
    # the predicted Mask R-CNN mask in the GrabCut mask
    rcnnMask = np.zeros(image.shape[:2], dtype="uint8")
    rcnnMask[startY:endY, startX:endX] = mask
    # apply a bitwise AND to the input image to show the output
    # of applying the Mask R-CNN mask to the image
    rcnnOutput = cv.bitwise_and(image, image, mask=rcnnMask)
    # show the output of the Mask R-CNN and bitwise AND operation
    cv.imshow("R-CNN Mask", rcnnMask)
    cv.imshow("R-CNN Output", rcnnOutput)
    cv.waitKey(0)

    # clone the Mask R-CNN mask (so we can use it when applying
    # GrabCut) and set any mask values greater than zero to be
    # "probable foreground" (otherwise they are "definite
    # background")
    gcMask = rcnnMask.copy()
    gcMask[gcMask > 0] = cv.GC_PR_FGD
    gcMask[gcMask == 0] = cv.GC_BGD
    # allocate memory for two arrays that the GrabCut algorithm
    # internally uses when segmenting the foreground from the
    # background and then apply GrabCut using the mask
    # segmentation method
    print("[INFO] applying GrabCut to '{}' ROI...".format(LABELS[classID]))
    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")
    (gcMask, bgModel, fgModel) = cv.grabCut(image, gcMask,None, bgModel, fgModel, iterCount=args["iter"],
                                             mode=cv.GC_INIT_WITH_MASK)

    # set all definite background and probable background pixels
    # to 0 while definite foreground and probable foreground
    # pixels are set to 1, then scale the mask from the range
    # [0, 1] to [0, 255]
    outputMask = np.where((gcMask == cv.GC_BGD) | (gcMask == cv.GC_PR_BGD), 0, 1)
    outputMask = (outputMask * 255).astype("uint8")
    # apply a bitwise AND to the image using our mask generated
    # by GrabCut to generate our final output image
    output = cv.bitwise_and(image, image, mask=outputMask)
    # show the output GrabCut mask as well as the output of
    # applying the GrabCut mask to the original input image
    cv.imshow("GrabCut Mask", outputMask)
    cv.imshow("Output", output)
    cv.waitKey(0)


    # Contour detection method
    import random as rng
    rng.seed(12345)

#         imgray = cv.cvtColor(outputMask, cv.COLOR_BGR2GRAY)
#         ret, thresh = cv.threshold(outputMask, 127, 255, 0)
    ret, thresh = cv.threshold(rcnnMask, 127, 255, 0)


    # Detect edges using Canny
#         canny_output = cv.Canny(outputMask, 100, 200)
    canny_output = cv.Canny(rcnnMask, 100, 200)

    #Find contours for detected edges
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    ratios = []
    areas = []
    circumferences = []        

    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
#             cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
        cv.drawContours(image, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
        area = cv.contourArea(contours[i])
        circumference = cv.arcLength(contours[i],True)
        if area > 0:
            ratio_area_to_circumference = area/circumference
            if ratio_area_to_circumference >= 5:
                ratios.append(ratio_area_to_circumference)
                areas.append(area)
                circumferences.append(circumference)
            print("ratio {} and iter->{}".format(ratio_area_to_circumference,i))
        print("area {} and i->{}".format(area,i))
        print("circumference {} and i->{}".format(circumference,i))
        print("--------------------------------")
    # Show in a window
    print("Ratios {}".format(ratios))
    header = ['imagepath', 'is_flying', 'ratio', 'area', 'circumference']
    data = []
    for i in range(len(ratios)):
        if ratios[i] < 19:
            #label 1: is_flying
            data.append(['is_flying', 1, ratios[i], areas[i], circumferences[i]])
#                 imagepath, 1(is_flying), ratios[i], areas[i], circumferences[i]
        else:
            #label 0: is_flying(not flying)
            data.append(['is_not_flying', 0, ratios[i], areas[i], circumferences[i]])
#                 imagepath, 0(is_flying), ratios[i], areas[i], circumferences[i]
    write_to_csv('bird_ratio.csv',None,data)
#         cv.imshow('Contours', drawing)
    cv.imshow('Contours', image)
    cv.waitKey(0)

    #Fourier Transform
    #crop the image as startY:endY, startX:endX 
    bb_image = output[startY:endY, startX:endX]
    bb_image = cv.cvtColor(bb_image, cv.COLOR_BGR2GRAY)
    try:
        #Apply Fourier Transform on the masked image
        fourier = np.fft.fft2(bb_image)
        fourier_shift = np.fft.fftshift(fourier)
        magnitude_spectrum = 20*np.log(np.abs(fourier_shift))
        magnitude_spectrum = np.asarray(magnitude_spectrum, dtype=np.uint8)
        # axis-1, we want to concatenate them horizontally 
        bb_img_and_magnitude = np.concatenate((bb_image, magnitude_spectrum), axis=0)

    except:
        print("Unable to obtain fourier transform for {} ".format(image_dict['path']))

    cv.imshow("Cropped Image",bb_img_and_magnitude)
    cv.waitKey(0)


    # Detect SIFT Features
#     img = output
    img = image[startY:endY, startX:endX]
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp = sift.detect(gray,None)
    img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #Compute the descriptors from the keypoints 
    kp,des = sift.compute(gray,kp)
    print(des)
    print(len(des))
    # Select random 50 keypoint descriptors 
    descriptors = des[np.random.randint(des.shape[0], size = 50)]
    print(len(descriptors))
    cv.imshow('sift_keypoints.jpg',img)
    cv.waitKey(0)
        
        

    

# # loop over the number of detected objects
# for i in range(0, boxes.shape[2]):
#     # extract the class ID of the detection along with the
#     # confidence (i.e., probability) associated with the
#     # prediction
#     classID = int(boxes[0, 0, i, 1])
#     confidence = boxes[0, 0, i, 2]
#     # filter out weak predictions by ensuring the detected
#     # probability is greater than the minimum probability
#     if confidence > args["confidence"]:
#         # show the class label
#         print("[INFO] showing output for '{}'...".format(LABELS[classID]))
#         # scale the bounding box coordinates back relative to the
#         # size of the image and then compute the width and the
#         # height of the bounding box
#         (H, W) = image.shape[:2]
#         box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
#         (startX, startY, endX, endY) = box.astype("int")
#         boxW = endX - startX
#         boxH = endY - startY
        
#         # extract the pixel-wise segmentation for the object, resize
#         # the mask such that it's the same dimensions as the bounding
#         # box, and then finally threshold to create a *binary* mask
#         mask = masks[i, classID]
#         mask = cv.resize(mask, (boxW, boxH),interpolation=cv.INTER_CUBIC)
#         mask = (mask > args["threshold"]).astype("uint8") * 255
#         # allocate a memory for our output Mask R-CNN mask and store
#         # the predicted Mask R-CNN mask in the GrabCut mask
#         rcnnMask = np.zeros(image.shape[:2], dtype="uint8")
#         rcnnMask[startY:endY, startX:endX] = mask
#         # apply a bitwise AND to the input image to show the output
#         # of applying the Mask R-CNN mask to the image
#         rcnnOutput = cv.bitwise_and(image, image, mask=rcnnMask)
#         # show the output of the Mask R-CNN and bitwise AND operation
#         cv.imshow("R-CNN Mask", rcnnMask)
#         cv.imshow("R-CNN Output", rcnnOutput)
#         cv.waitKey(0)
        
#         # clone the Mask R-CNN mask (so we can use it when applying
#         # GrabCut) and set any mask values greater than zero to be
#         # "probable foreground" (otherwise they are "definite
#         # background")
#         gcMask = rcnnMask.copy()
#         gcMask[gcMask > 0] = cv.GC_PR_FGD
#         gcMask[gcMask == 0] = cv.GC_BGD
#         # allocate memory for two arrays that the GrabCut algorithm
#         # internally uses when segmenting the foreground from the
#         # background and then apply GrabCut using the mask
#         # segmentation method
#         print("[INFO] applying GrabCut to '{}' ROI...".format(LABELS[classID]))
#         fgModel = np.zeros((1, 65), dtype="float")
#         bgModel = np.zeros((1, 65), dtype="float")
#         (gcMask, bgModel, fgModel) = cv.grabCut(image, gcMask,None, bgModel, fgModel, iterCount=args["iter"],
#                                                  mode=cv.GC_INIT_WITH_MASK)
        
#         # set all definite background and probable background pixels
#         # to 0 while definite foreground and probable foreground
#         # pixels are set to 1, then scale the mask from the range
#         # [0, 1] to [0, 255]
#         outputMask = np.where((gcMask == cv.GC_BGD) | (gcMask == cv.GC_PR_BGD), 0, 1)
#         outputMask = (outputMask * 255).astype("uint8")
#         # apply a bitwise AND to the image using our mask generated
#         # by GrabCut to generate our final output image
#         output = cv.bitwise_and(image, image, mask=outputMask)
#         # show the output GrabCut mask as well as the output of
#         # applying the GrabCut mask to the original input image
#         cv.imshow("GrabCut Mask", outputMask)
#         cv.imshow("Output", output)
#         cv.waitKey(0)


