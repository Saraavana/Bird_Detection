import pandas as pd
import cv2 as cv
import imutils
import helper
import numpy as np
import random as rng
from utils.nontf_util import filter_bounding_boxes
from sklearn.metrics import confusion_matrix
import sklearn.metrics

import wandb
from wandb.keras import WandbCallback

from sklearn.model_selection import train_test_split

def initiate_wandb(project,name):
    # 1. Initialize a new wandb run
    run = wandb.init(project=project, 
                entity='elsaravana', 
                name=name)
    if run is None:
        raise ValueError("Wandb didn't initialize properly")


def compute_confusion_matrix(dataframe):
    pred = dataframe['pred_is_flying'].astype(float).astype(int).to_numpy()
    true = dataframe['truth_is_flying'].astype(int).to_numpy()

    tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
    print(tn, fp, fn, tp)
    wandb.log({"confusion_matrix_threshold_19" : wandb.plot.confusion_matrix(probs=None,
                            preds=pred, y_true=true,
                            class_names=["is_not_flying","is_flying"])})
    wandb.log({"precision vs recall" : wandb.plot.pr_curve(true, pred,
                     labels=None, classes_to_plot=None)})

# dataframee = load('3k_images_from_ratio.csv')
# print(dataframee['pred_is_flying'].astype(float).astype(int).to_numpy())

#     image = cv.resize(image, (400, 400), interpolation = cv.INTER_AREA)
# 151. 003.75 - Shape is (217, 500, 3) = AR: 0.434
# 2517. 44.0006 - Shape is (335, 500, 3) = AR: 0.67
# 1037 - 2809. 49.100 - Shape is (375, 500, 3) = AR: 0.75
# 1980 - 5341. 92.61 - Shape is (500, 341, 3) = AR: 341/500 = 0.682
# 2660. 119.0125 - Shape is (375, 500, 3) = AR: 0.75

# scale = 1.53
# 2309. 105.0046 - Shape is (334, 500, 3) = AR: 0.668
# 2316. 105.0018 - Shape is (326, 500, 3) = AR: 0.652
# 2320. 105.Whip_poor_Will/Whip_Poor_Will_0001_796411.jpg - Shape is (164, 230, 3) = AR: 0.713
# 2660. 119.Field_Sparrow/Field_Sparrow_0125_113869.jpg - Shape is (375, 500, 3) = AR: 0.75

def depth_to_distance(depth):
    return -1.7 * depth + 2

def evaluate_depth_map(img_path, startX, startY, endX, endY):
    image = cv.imread(img_path)
    scale_up = 1.53
    image = cv.resize(image, None, fx= scale_up, fy= scale_up, interpolation= cv.INTER_LINEAR)
    imgHeight, imgWidth, channel = image.shape

    # Load the DNN model
    # model = cv.dnn.readNet(helper.midas_large_path)
    model = cv.dnn.readNet(helper.midas_small_path)

    if (model.empty()):
        print("Could not load the neural net! - Check path")

    # Convert the BGR image to RGB
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    # Estimate the depth of the bounding box from camera
    # Center point of the detected bounding box 
    print(startX, startY, endX, endY)
    center_point = ((startX + endX) / 2, (startY + endY) / 2)

    # blob = cv.dnn.blobFromImage(image, True, False)
    blob = cv.dnn.blobFromImage(image, 1/255., (256,256), (123.675, 116.28, 103.53), True, False)
    # blob = cv.dnn.blobFromImage(image, swapRB=True, crop=False)

    # MiDaS v2.1 Small ( Scale : 1 / 255, Size : 256 x 256, Mean Subtraction : ( 123.675, 116.28, 103.53 ), Channels Order : RGB )
    #blob = cv2.dnn.blobFromImage(img, 1/255., (256,256), (123.675, 116.28, 103.53), True, False)

    # Set input to the model
    model.setInput(blob)

    # Make forward pass in model
    depth_map = model.forward()
    
    depth_map = depth_map[0,:,:]

    depth_map = cv.resize(depth_map, (imgWidth, imgHeight))

    # Normalize the output
    depth_map = cv.normalize(depth_map, None, 0, 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    # depth_map = cv.normalize(depth_map, None, 0, 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
    # depth_map = cv.normalize(depth_map, None, 0, 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32S)

    # Convert the image color back so it can be displayed
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    ## ----------------------------------------------------------------------------------------
    # Depth to bird
    
    depth_bird = depth_map[int(center_point[1]), int(center_point[0])]
    # print("Bird depth to before distance estimation: ", depth_bird)
    depth_bird = depth_to_distance(depth_bird)
    
    return depth_map, depth_bird



def ratio_estimation():

    ###Contour detection using RNN Mask 

    threek_sample_df = helper.load_data_for_ratio_method().iloc[:3000]
    # threek_sample_df = helper.load_data_for_ratio_method().iloc[:3]
    # threek_sample_df = helper.load_data_for_ratio_method().iloc[1942:1946]

    # threek_sample_df = helper.load_data_for_ratio_method().iloc[:5]   
    # threek_sample_df = helper.load_data_for_ratio_method().iloc[2996:3001]

    threek_sample_df = threek_sample_df.drop(columns=['sift_keypoints', 'is_training'])

    # condition = threek_sample_df['truth_is_flying'] == '0'
    # threek_sample_df = threek_sample_df[condition]
    # threek_sample_df = threek_sample_df.iloc[:302]
    # print("The number of flying birds - ",len(threek_sample_df))

    iters = 1
    # thresh = float(22)
    # thresh = thres_value

    for img_name in threek_sample_df['image_name']:
        img_path = helper.dataset_path + "images/" + img_name
        img_path = img_path.strip()

        # load our input image from disk and display it to our screen
        image = cv.imread(img_path)

    #     image = cv.resize(image, (400, 400), interpolation = cv.INTER_AREA)
        # Scaling up the image 1.53 times specifying a single scale factor.
        scale_up = 1.53
        image = cv.resize(image, None, fx= scale_up, fy= scale_up, interpolation= cv.INTER_LINEAR)
        print("Image name..{}".format(img_name))
        # cv.imshow("Input", image)
        # cv.waitKey(0)

        # construct a blob from the input image and then perform a
        # forward pass of the Mask R-CNN, giving us (1) the bounding box
        # coordinates of the objects in the image along with (2) the
        # pixel-wise segmentation for each specific object
        blob = cv.dnn.blobFromImage(image, swapRB=True, crop=False)
        model = helper.load_model()
        model.setInput(blob)
        (boxes, masks) = model.forward(["detection_out_final","detection_masks"])
        
        # # Make forward pass in model
        # depth_map = model.forward()

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
            
            # result = image.copy()
            # red = (0, 0, 255)
            # thickness = 2
            # cv.rectangle(result, (startX, startY), (startX+boxW, startY+boxH), red, thickness)

            # depth_map, depth_bird = evaluate_depth_map(img_path, startX, startY, endX, endY)

            # print("Distance to bird:",depth_bird)
            # depth_bird = round(depth_bird,2)
            # cv.putText(result, "Depth : " + str(depth_bird) + "m", (50,400), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)

            mask_ratio = helper.annotate_largest_contour(rcnnMask,image,boxW,boxH)

            # is_flying = None
            # if mask_ratio <= thresh:
            #     #label - is_flying - 1
            #     is_flying = 1
            # else:
            #     #label - is_flying - 0
            #     is_flying = 0


            # prob_of_non_flying = None
            # prob_of_flying = None

            # if mask_ratio > thresh:
            #     prob_of_non_flying = thresh/mask_ratio
            #     prob_of_flying = 1 - prob_of_non_flying
            # else:
            #     prob_of_flying = mask_ratio/thresh
            #     prob_of_non_flying = 1 - prob_of_flying
            
            # probas = np.array([prob_of_non_flying, prob_of_flying])
            
            # threek_sample_df.at[iters, 'pred_probas'] = probas
            # threek_sample_df.at[iters, 'ratio'] = mask_ratio
            # threek_sample_df.at[iters, 'pred_is_flying'] = is_flying

            # threek_sample_df.loc[threek_sample_df['image_name']==img_name, 'pred_probas'] = probas
            threek_sample_df.loc[threek_sample_df['image_name']==img_name, 'ratio'] = mask_ratio
            # threek_sample_df.loc[threek_sample_df['image_name']==img_name, 'pred_is_flying'] = is_flying            
            
            # save resulting image
            # cv.imwrite('bounding_box_result.jpg',result)      
            print("The ratio is ",mask_ratio)

            # cv.imshow("Input", result)
            # cv.imshow("Depth map", depth_map)
            # cv.waitKey(0)

        print("Iteration no.",iters)
        iters = iters+1
        # threek_sample_df.to_csv('3k_images_from_ratio.csv')
    # threek_sample_df.to_json('3k_images_from_ratio_31.json')

    # threek_sample_df.to_json('3k_images_from_ratio_22.json')
    # threek_sample_df.to_json('3k_images_from_ratio_23.json')
    # threek_sample_df.to_json('3k_images_from_ratio_24.json')
    # threek_sample_df.to_json('3k_images_from_ratio_25.json')
    # threek_sample_df.to_json('3k_images_from_ratio_26.json')
    # threek_sample_df.to_json('3k_images_from_ratio_27.json')
    threek_sample_df.to_json(json_name)

    # threek_sample_df.to_json('only_flying_birds_30.json')
    # threek_sample_df.to_json('only_non_flying_birds_25.json')
    # threek_sample_df.to_json('only_non_flying_birds_30.json')


json_name = "ratio_values_for_3k_images.json"

# ratio_estimation()

def ratio_metrics():
    thresholds = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,31,32,33,34,35,36,37,38,39,40]
    # thresholds = [14,14.10,14.20,14.30,14.40,14.50,14.60,14.70,14.80,14.90,15,15.10,15.20,15.30,15.40,15.50,15.60,15.70,15.80,15.90,16]
    # thresholds = [14.70,14.71,14.72,14.73,14.74,14.75,14.76,14.77,14.78,14.79,14.80,14.81,14.82,14.83,14.84,14.85,14.86,14.87,14.88,14.89,14.90,14.91,14.92,14.93,14.94,14.95,14.96,14.97,14.98,14.99,15]
    # thresholds = [14.90,14.91,14.92,14.93,14.94,14.95,14.96,14.97,14.98,14.99,15,15.01,15.02,15.03,15.04,15.05,15.06,15.07,15.08,15.09,15.10,15.11,15.12,15.13,15.14,15.15,15.16,15.17,15.18,15.19,15.20]
    # thresholds = [21,21.10,21.20,21.30,21.40,21.50,21.60,21.70,21.80,21.90,22]
    # thresholds = [21.90,21.91,21.92,21.93,21.94,21.95,21.96,21.97,21.98,21.99,22,22.01,22.02,22.03,22.04,22.05,22.06,22.07,22.08,22.09,22.10]


    df = pd.read_json(json_name)
    df1 = df.copy()
    # train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    # df1 = test_df.copy()
    print("Lenght of Dataframe: ",df1.shape[0])
    f1 = []

    for each_thresh in thresholds:
        print("Threshold: ",each_thresh)
        print("-------------------------")
        df1.loc[df1['ratio'] <= each_thresh, 'pred'] = 1
        df1.loc[df1['ratio'] > each_thresh, 'pred'] = 0
        pred = df1['pred']
        truth = df1['truth_is_flying']
        accuracy = sklearn.metrics.accuracy_score(truth, pred)
        precision = sklearn.metrics.precision_score(truth, pred)
        recall = sklearn.metrics.recall_score(truth, pred)
        f1_measure = sklearn.metrics.f1_score(truth, pred)
        f1.append(f1_measure)
        print("Accuracy: ", accuracy, "Precision: ", precision, "Recall: ", recall, "F1_score: ", f1_measure)
        print("-------------------------")

    f1_np = np.array(f1)
    f1_max = np.max(f1_np)
    print("Maximum F1 Score: ", f1_max)
    print("Maximum Threshold: ",thresholds[f1_np.argmax()])

ratio_metrics()

# Iteration on 3000 images:
# -------------------------
# Threshold:  15
# -------------------------
# Accuracy:  0.7546666666666667 Precision:  0.1873198847262248 Recall:  0.4304635761589404 F1_score:  0.26104417670682734
# -------------------------
# Maximum F1 Score:  0.26104417670682734
# -------------------------------------------------------------------
# -------------------------
# Optimal Threshold:  15.02
# -------------------------
# Accuracy:  0.7546666666666667 Precision:  0.1882183908045977 Recall:  0.4337748344370861 F1_score:  0.2625250501002004
# -------------------------
# Optimal Maximum F1 Score:  0.2625250501002004

# Iteration on 600 images:
# -------------------------
# Threshold:  15
# -------------------------
# Accuracy:  0.7533333333333333 Precision:  0.16428571428571428 Recall:  0.42592592592592593 F1_score:  0.2371134020618557
# -------------------------
# Maximum F1 Score:  0.2371134020618557
# -------------------------------------------------------------------
# -------------------------
# Optimal Threshold:  14.74
# -------------------------
# Accuracy:  0.7566666666666667 Precision:  0.16666666666666666 Recall:  0.42592592592592593 F1_score:  0.23958333333333334
# -------------------------
# Optimal Maximum F1 Score:  0.23958333333333334


# initiate_wandb('Binary-classification-with-ratio-method','bird-ratio-classifier_31')
# initiate_wandb('Binary-classification-with-ratio-method','bird-ratio-classifier_25')
# compute_confusion_matrix(helper.load_from('3k_images_from_ratio.csv'))

# df = pd.read_json('3k_images_from_ratio_31.json')

# df = pd.read_json('only_flying_birds_30.json')
# df = pd.read_json('only_flying_birds_25.json')
# df = pd.read_json('only_non_flying_birds_25.json')
# df = pd.read_json('only_non_flying_birds_30.json')
# # pred_probas = df['pred_probas']
# true = df['truth_is_flying']
# pred = df['pred_is_flying']


# condition = df['pred_is_flying'] == 0
# tn_df = df[condition]  

# print(tn_df['image_name'][6])
# for each in tn_df['ratio']:
#     print(each)

# tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
# print(tn, fp, fn, tp)

# true = np.array(true)
# pred = np.array(pred)

# probas = []

# for each in pred_probas:
#     probas.append(np.array(each))

# probas = np.array(probas)

# wandb.log({"confusion_matrix_threshold_25" : wandb.plot.confusion_matrix(probs=None, preds=pred, y_true=true, class_names=["is_not_flying","is_flying"])})
# wandb.log({"precision vs recall" : wandb.plot.pr_curve(true, probas, labels=["is_not_flying","is_flying"], classes_to_plot=None)})
# wandb.log({"ROC Curve" : wandb.plot.roc_curve(true, probas, labels=["is_not_flying","is_flying"], classes_to_plot=None)})



# 1. record probas - [probability in 0 class, probability in 1 class]
# 2. record ratios


# Threshold - 19
# tn, fp, fn, tp = 2679, 19, 286, 16
# Precision - 0.457
# Recall - 0.053
# F1 Measure - 0.095

# Threshold - 20
# tn, fp, fn, tp = 2668, 30, 281, 21 
# Precision - 0.412
# Recall - 0.070
# F1 Measure - 0.119

# Threshold - 21
# tn, fp, fn, tp = 2654, 44, 278, 24
# Precision - 0.353
# Recall - 0.079
# F1 Measure - 0.130

# Threshold - 22
# tn, fp, fn, tp = 2644, 54, 274, 28
# Precision - 0.341
# Recall - 0.093
# F1 Measure - 0.146

# Threshold - 23
# tn, fp, fn, tp = 2622, 76, 269, 33
# Precision - 0.303
# Recall - 0.109
# F1 Measure - 0.161

# Threshold - 24
# tn, fp, fn, tp = 2600, 98, 259, 43
# Precision - 0.305
# Recall - 0.142
# F1 Measure - 0.194

# Threshold - 25
# tn, fp, fn, tp = 2563, 135, 244, 58
# Precision - 0.301
# Recall - 0.192
# F1 Measure - 0.234

# Threshold - 26
# tn, fp, fn, tp = 2524, 174, 234, 68
# Precision - 0.281
# Recall - 0.225
# F1 Measure - 0.250

# Threshold - 27
# tn, fp, fn, tp = 2486, 212, 224, 78
# Precision - 0.269
# Recall - 0.258
# F1 Measure - 0.264

# Threshold - 28
# tn, fp, fn, tp = 2443, 255, 210, 92
# Precision - 0.265
# Recall - 0.305
# F1 Measure - 0.284

# Threshold - 29
# tn, fp, fn, tp = 2402, 296, 200, 102
# Precision - 0.256
# Recall - 0.338
# F1 Measure - 0.291

# Threshold - 30
# tn, fp, fn, tp = 2354, 344, 195, 107
# Precision - 0.237
# Recall - 0.354
# F1 Measure - 0.284


#Test from 3000 image samples
# ----------------------------------------------------------------------------------------------------------------

# Test 1:

# All the bird bounding detection from 500 images are allowed for contour estimation, only one bounding box per image
# with highest confidence score of detections is considered, contour is drawed on original image, 
# it yield [482] contour images. Each of these contour images are of dim(400,400) has contour ratio => 5, 
# and if it is <= 26, labeled as 'flying', if > 26 then 'not-flying'. 

# No. of non-flying birds labelled as non-flying(True Negative) - 274/500
# No. of non-flying birds labelled as flying(False Positive) - 166/500
# No. of flying birds labelled as non-flying(False Negative) - 10/500
# No. of flying birds labelled as flying(True Positive) - 50/500


# Test 2:

# All the bird bounding detection from 500 images are allowed for contour estimation, only one bounding box per image
# with highest confidence score of detections is considered, contour is drawed on original image, 
# it yield [482] contour images. Each of these contour images are of dim(400,400) has contour ratio => 5, 
# and if it is <= 27, labeled as 'flying', if > 27 then 'not-flying'. 

# No. of non-flying birds labelled as non-flying(True Negative) - 257/500
# No. of non-flying birds labelled as flying(False Positive) - 183/500
# No. of flying birds labelled as non-flying(False Negative) - 9/500
# No. of flying birds labelled as flying(True Positive) - 51/500


# Test 3:

# All the bird bounding detection from 500 images are allowed for contour estimation, only one bounding box per image
# with highest confidence score of detections is considered, contour is drawed on original image, 
# it yield [482] contour images. Each of these contour images are of dim(400,400) has contour ratio => 5, 
# and if it is <= 28, labeled as 'flying', if > 28 then 'not-flying'. 

# No. of non-flying birds labelled as non-flying(True Negative) - 245/500
# No. of non-flying birds labelled as flying(False Positive) - 195/500
# No. of flying birds labelled as non-flying(False Negative) - 6/500
# No. of flying birds labelled as flying(True Positive) - 54/500


# Test 4:

# All the bird bounding detection from 500 images are allowed for contour estimation, only one bounding box per image
# with highest confidence score of detections is considered, contour is drawed on original image, 
# it yield [482] contour images. Each of these contour images are of dim(400,400) has contour ratio => 5, 
# and if it is <= 29, labeled as 'flying', if > 29 then 'not-flying'. 

# No. of non-flying birds labelled as non-flying(True Negative) - 229/500
# No. of non-flying birds labelled as flying(False Positive) - 211/500
# No. of flying birds labelled as non-flying(False Negative) - 5/500
# No. of flying birds labelled as flying(True Positive) - 55/500


# Test 5:

# All the bird bounding detection from 500 images are allowed for contour estimation, only one bounding box per image
# with highest confidence score of detections is considered, contour is drawed on original image, 
# it yield [482] contour images. Each of these contour images are of dim(400,400) has contour ratio => 5, 
# and if it is <= 30, labeled as 'flying', if > 30 then 'not-flying'. 

# No. of non-flying birds labelled as non-flying(True Negative) - 210/500
# No. of non-flying birds labelled as flying(False Positive) - 230/500
# No. of flying birds labelled as non-flying(False Negative) - 3/500
# No. of flying birds labelled as flying(True Positive) - 57/500


# Test 6:

# All the bird bounding detection from 500 images are allowed for contour estimation, only one bounding box per image
# with highest confidence score of detections is considered, contour is drawed on original image, 
# it yield [482] contour images. Each of these contour images are of dim(400,400) has contour ratio => 5, 
# and if it is <= 25, labeled as 'flying', if > 25 then 'not-flying'. 

# No. of non-flying birds labelled as non-flying(True Negative) - 293/500
# No. of non-flying birds labelled as flying(False Positive) - 147/500
# No. of flying birds labelled as non-flying(False Negative) - 15/500
# No. of flying birds labelled as flying(True Positive) - 45/500


# Test 7:

# All the bird bounding detection from 500 images are allowed for contour estimation, only one bounding box per image
# with highest confidence score of detections is considered, contour is drawed on original image, 
# it yield [482] contour images. Each of these contour images are of dim(400,400) has contour ratio => 5, 
# and if it is <= 24, labeled as 'flying', if > 24 then 'not-flying'.  

# No. of non-flying birds labelled as non-flying(True Negative) - 312/500
# No. of non-flying birds labelled as flying(False Positive) - 128/500
# No. of flying birds labelled as non-flying(False Negative) - 18/500
# No. of flying birds labelled as flying(True Positive) - 42/500


# Test 8:

# All the bird bounding detection from 500 images are allowed for contour estimation, only one bounding box per image
# with highest confidence score of detections is considered, contour is drawed on original image, 
# it yield [482] contour images. Each of these contour images are of dim(400,400) has contour ratio => 5, 
# and if it is <= 23, labeled as 'flying', if > 23 then 'not-flying'. 

# No. of non-flying birds labelled as non-flying(True Negative) - 327/500
# No. of non-flying birds labelled as flying(False Positive) - 113/500
# No. of flying birds labelled as non-flying(False Negative) - 20/500
# No. of flying birds labelled as flying(True Positive) - 40/500


# Test 9:

# All the bird bounding detection from 500 images are allowed for contour estimation, only one bounding box per image
# with highest confidence score of detections is considered, contour is drawed on original image, 
# it yield [482] contour images. Each of these contour images are of dim(400,400) has contour ratio => 5, 
# and if it is <= 22, labeled as 'flying', if > 22 then 'not-flying'. 

# No. of non-flying birds labelled as non-flying(True Negative) - 343/500
# No. of non-flying birds labelled as flying(False Positive) - 97/500
# No. of flying birds labelled as non-flying(False Negative) - 26/500
# No. of flying birds labelled as flying(True Positive) - 34/500

# Test 10:

# All the bird bounding detection from 500 images are allowed for contour estimation, only one bounding box per image
# with highest confidence score of detections is considered, contour is drawed on original image, 
# it yield [482] contour images. Each of these contour images are of dim(400,400) has contour ratio => 5, 
# and if it is <= 21, labeled as 'flying', if > 21 then 'not-flying'. 

# No. of non-flying birds labelled as non-flying(True Negative) - 351/500
# No. of non-flying birds labelled as flying(False Positive) - 89/500
# No. of flying birds labelled as non-flying(False Negative) - 29/500
# No. of flying birds labelled as flying(True Positive) - 31/500

# Test 11:

# All the bird bounding detection from 500 images are allowed for contour estimation, only one bounding box per image
# with highest confidence score of detections is considered, contour is drawed on original image, 
# it yield [482] contour images. Each of these contour images are of dim(400,400) has contour ratio => 5, 
# and if it is <= 20, labeled as 'flying', if > 20 then 'not-flying'. 

# No. of non-flying birds labelled as non-flying(True Negative) - 367/500
# No. of non-flying birds labelled as flying(False Positive) - 73/500
# No. of flying birds labelled as non-flying(False Negative) - 29/500
# No. of flying birds labelled as flying(True Positive) - 31/500



