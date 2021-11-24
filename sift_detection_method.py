import numpy as np
import helper
import cv2 as cv
from utils.nontf_util import filter_bounding_boxes

import wandb
from wandb.keras import WandbCallback

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import pandas as pd
import matplotlib.pyplot as plt

def initiate_wandb(project,name):
    # 1. Initialize a new wandb run
    run = wandb.init(project=project, 
                entity='elsaravana', 
                name=name)
    if run is None:
        raise ValueError("Wandb didn't initialize properly")


def sift_detection():
    ###Contour detection using (RNN Mask) 
    threek_sample_df = helper.load_3k_images_from('ratio_method_images.csv').iloc[:3000]
    iters = 1
    
    for img_name in threek_sample_df['image_name']:
        img_path = helper.dataset_path + "images/" + img_name
        img_path = img_path.strip()

        # load our input image from disk and display it to our screen
        image = cv.imread(img_path)
        # image = cv.resize(image, (400, 400), interpolation = cv.INTER_AREA)
        # Scaling up the image 1.53 times specifying a single scale factor.
        scale_up = 1.53
        image = cv.resize(image, None, fx= scale_up, fy= scale_up, interpolation= cv.INTER_LINEAR)


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

        if len(filtered_boxes) > 0:
            # scale the bounding box coordinates back relative to the
            # size of the image and then compute the width and the
            # height of the bounding box
            (H, W) = image.shape[:2]
            box = filtered_boxes[0, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY

            #crop the image as startY:endY, startX:endX 
            bb_image = image[startY:endY, startX:endX]

            # Detect SIFT Features
            gray= cv.cvtColor(bb_image,cv.COLOR_BGR2GRAY)
            sift = cv.SIFT_create()
            kp = sift.detect(gray,None)
            #Compute the descriptors from the keypoints 
            kp,des = sift.compute(gray,kp)
            
            # Select random 50 keypoint descriptors 
            descriptors = des[np.random.randint(des.shape[0], size = 50)]
            
            # threek_sample_df.loc[threek_sample_df['image_name']==img_name, 'sift_keypoints'] = descriptors
            threek_sample_df.at[iters, 'sift_keypoints'] = descriptors
            
        
        print("Iteration no.",iters)
        iters = iters+1     
        
    # threek_sample_df.to_csv('3k_images_with_SIFT.csv')
    threek_sample_df.to_json('3k_images_with_SIFT.json')


def linear_kernal_SVC():
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    
    # Log confusion matrix
    # the key "confusion_matrix_linear_svc" is the id of the plot--do not change
    # this if you want subsequent runs to show up on the same plot
    wandb.log({"confusion_matrix_linear_svc" : wandb.plot.confusion_matrix(probs=None,
                            y_true=y_test, preds=y_pred,
                            class_names=["is_not_flying","is_flying"])})


def poly_kernal_SVC():
    svclassifier = SVC(kernel='poly', degree=8)
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    # Log confusion matrix
    # the key "confusion_matrix_polynomial_svc" is the id of the plot--do not change
    # this if you want subsequent runs to show up on the same plot
    wandb.log({"confusion_matrix_polynomial_svc" : wandb.plot.confusion_matrix(probs=None,
                            preds=y_pred, y_true=y_test,
                            class_names=["is_not_flying","is_flying"])})

def gaussian_kernal_SVC():
    svclassifier = SVC(kernel='rbf')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    # Log confusion matrix
    # the key "confusion_matrix_gaussian_svc" is the id of the plot--do not change
    # this if you want subsequent runs to show up on the same plot
    wandb.log({"confusion_matrix_gaussian_svc" : wandb.plot.confusion_matrix(probs=None,
                            preds=y_pred, y_true=y_test,
                            class_names=["is_not_flying","is_flying"])})

def sigmoid_kernal_SVC():
    svclassifier = SVC(kernel='sigmoid')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    # Log confusion matrix
    # the key "confusion_matrix_sigmoid_svc" is the id of the plot--do not change
    # this if you want subsequent runs to show up on the same plot
    wandb.log({"confusion_matrix_sigmoid_svc" : wandb.plot.confusion_matrix(probs=None,
                            preds=y_pred, y_true=y_test,
                            class_names=["is_not_flying","is_flying"])})



# sift_detection()


# Classification & Evaluation 

df = pd.read_json('3k_images_with_SIFT.json')
sift_data = df['sift_keypoints']
y_labels = df['truth_is_flying']

x_descriptors = []

for index, row in sift_data.iteritems():
    each_sift_descriptor = np.array(sift_data[index], dtype=float)
    x_descriptors.append(each_sift_descriptor)

dim3_X_values = np.array(x_descriptors)
nsamples, nx, ny = dim3_X_values.shape
dim2_X_values = dim3_X_values.reshape((nsamples,nx*ny))




# X_train, X_test, y_train, y_test = train_test_split(dim2_X_values, y_labels, test_size = 0.20, random_state=42)
# y_test = np.array(y_test)
# initiate_wandb('Binary-classification-with-sift-features','k-means-clustering')


# Sum_of_squared_distances = []
# K = range(1,10)
# K = [2,3,4,5,6,7,8,9,10]

# for num_clusters in K :
#     print("Started with cluster no: ",num_clusters)
#     kmeans = KMeans(n_clusters=num_clusters)
#     kmeans.fit(dim2_X_values)
#     Sum_of_squared_distances.append(kmeans.inertia_)

# plt.plot(K,Sum_of_squared_distances,'bx-')
# plt.xlabel('Values of K') 
# plt.ylabel('Sum of squared distances/Inertia') 
# plt.title('Elbow Method For Optimal k')
# plt.show()

# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
# range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
range_n_clusters = [2,3,4,5,6,7,8,9,10]
silhouette_avg = []

for num_clusters in range_n_clusters:
    # initialise kmeans
    print("Started with cluster no: ",num_clusters)
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(dim2_X_values)
    cluster_labels = kmeans.labels_
    
    # silhouette score
    silhouette_avg.append(silhouette_score(dim2_X_values, cluster_labels))
 
plt.plot(range_n_clusters,silhouette_avg,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Silhouette score') 
plt.title('Silhouette analysis For Optimal k')
plt.show()


# linear_kernal_SVC()
# poly_kernal_SVC()
# gaussian_kernal_SVC()
# sigmoid_kernal_SVC()