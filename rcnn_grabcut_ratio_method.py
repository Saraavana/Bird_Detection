import helper
import cv2 as cv
import pandas as pd
import numpy as np
from utils.nontf_util import filter_bounding_boxes
from sklearn.metrics import confusion_matrix

threek_sample_df = helper.load_data_for_ratio_method().iloc[:3000]

def compute_confusion_matrix():
    pred = threek_sample_df['pred_is_flying'].astype(int).to_numpy()
    true = threek_sample_df['truth_is_flying'].astype(int).to_numpy()

    tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
    print(tn, fp, fn, tp)

###Contour detection using (Grabcut + RNN Mask) 

iters = 0

for img_name in threek_sample_df['image_name']:
    img_path = helper.dataset_path + "images/" + img_name
    img_path = img_path.strip()

    # load our input image from disk and display it to our screen
    image = cv.imread(img_path)
    image = cv.resize(image, (400, 400), interpolation = cv.INTER_AREA)

    # construct a blob from the input image and then perform a
    # forward pass of the Mask R-CNN, giving us (1) the bounding box
    # coordinates of the objects in the image along with (2) the
    # pixel-wise segmentation for each specific object
    blob = cv.dnn.blobFromImage(image, swapRB=True, crop=False)
    model = helper.load_model()
    model.setInput(blob)
    (boxes, masks) = model.forward(["detection_out_final","detection_masks"])

    # Extract only the bounding boxes values only for 'bird' class
    # Convert the obtained f_boxes filter object of np.array
    f_boxes = filter(filter_bounding_boxes, boxes[0,0])
    filtered_boxes = np.array(list(f_boxes))

    # convert filtered bounding boxes of type np.array to list
    detected_boxes = boxes[0,0].tolist()

    
    # Perform non-maximum suppression on detected bounding boxes 
    filtered_boxes = helper.non_max_suppression_fast(filtered_boxes,0.50)  
    
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

        # filter out weak predictions by ensuring the detected
        # probability is greater than the minimum probability
#             if confidence > confidence_score:
            
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
        mask = (mask > helper.pixel_seg_threshold).astype("uint8") * 255
        # allocate a memory for our output Mask R-CNN mask and store
        # the predicted Mask R-CNN mask in the GrabCut mask
        rcnnMask = np.zeros(image.shape[:2], dtype="uint8")
        rcnnMask[startY:endY, startX:endX] = mask
        # apply a bitwise AND to the input image to show the output
        # of applying the Mask R-CNN mask to the image
        rcnnOutput = cv.bitwise_and(image, image, mask=rcnnMask)

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
        fgModel = np.zeros((1, 65), dtype="float")
        bgModel = np.zeros((1, 65), dtype="float")
        (gcMask, bgModel, fgModel) = cv.grabCut(image, gcMask,None, bgModel, fgModel, iterCount=helper.no_of_grabcut_iter,
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

        is_flying = helper.annotate_largest_contour(outputMask,image,19)

        threek_sample_df.loc[threek_sample_df['image_name']==img_name, "pred_is_flying"] = is_flying
            
    iters = iters+1      
    print("Iteration no.",iters)
threek_sample_df.to_csv('grabcut_ratio_method_3k_images.csv')



# compute_confusion_matrix()