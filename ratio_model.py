# Building a model
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.optimizers as Optimizer
import wandb
from wandb.keras import WandbCallback
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


def wandb_init(project,name):
    # 1. Start a new run
    run = wandb.init(
                project=project, 
                entity='elsaravana', 
                name=name,
                config={
                    "activation_1": "relu",
                    "activation_2":"sigmoid",
                    "loss": "binary_crossentropy",
                    "metric": "accuracy",
                    "epoch": 35,
                    "batch_size": 32
                })
    if run is None:
        raise ValueError("Wandb didn't initialize properly")

def cnn_model():
    # Defining our CNN Model Layers
    cnn_model = keras.Sequential()

    # Convolutional layer and maxpool layer 1
    cnn_model.add(keras.layers.Conv2D(32,(3,3),activation=config.activation_1,input_shape=(150,150,3)))
    cnn_model.add(keras.layers.MaxPool2D(2,2))

    # Convolutional layer and maxpool layer 2
    cnn_model.add(keras.layers.Conv2D(64,(3,3),activation=config.activation_1))
    cnn_model.add(keras.layers.MaxPool2D(2,2))

    # Convolutional layer and maxpool layer 3
    cnn_model.add(keras.layers.Conv2D(128,(3,3),activation=config.activation_1))
    cnn_model.add(keras.layers.MaxPool2D(2,2))

    # Convolutional layer and maxpool layer 4
    cnn_model.add(keras.layers.Conv2D(128,(3,3),activation=config.activation_1))
    cnn_model.add(keras.layers.MaxPool2D(2,2))

    # This layer flattens the resulting image array to 1D array
    cnn_model.add(keras.layers.Flatten())

    # Hidden layer with 512 neurons and Rectified Linear Unit activation function 
    cnn_model.add(keras.layers.Dense(512,activation=config.activation_1))

    # Output layer with single neuron which gives 0 for bird 'is_not_flying' or 1 for bird 'is_flying'
    #Here we use sigmoid activation function which makes our model output to lie between 0 and 1
    cnn_model.add(keras.layers.Dense(1,activation=config.activation_2))

    # If binary classification, then we use 'binary_crossentropy'. If it is multi-class classification
    # then we use 'sparse_categorical_crossentropy' as loss function
    cnn_model.compile(optimizer=Optimizer.Adam(lr=0.0001),loss=config.loss,metrics=[config.metric])

    cnn_model.summary()

    return cnn_model



wandb_init('Binary-classification-with-cnn-ratio','bird_ratio_classifier_20')
config = wandb.config

df = pd.read_json('3k_images_from_ratio_20.json')
train_df, test_df = train_test_split(df, test_size=0.2)

train_X = np.array(train_df['ratio'])
train_y = np.array(train_df['truth_is_flying'])

test_X = np.array(test_df['ratio'])
test_y = np.array(test_df['truth_is_flying'])

model = cnn_model()

model.fit(train_X, train_y, epochs = config.epoch, batch_size=config.batch_size, validation_split=0.30, callbacks=[WandbCallback()])

model.evaluate(test_X,test_y, verbose=1)

pred_probas = model.pred_probas(test_X)
preds = model.predict(test_X)


wandb.log({"confusion_matrix" : wandb.plot.confusion_matrix(probs=None, preds=preds, y_true=test_y, class_names=["is_not_flying","is_flying"])})
wandb.log({"precision vs recall" : wandb.plot.pr_curve(test_y, pred_probas, labels=["is_not_flying","is_flying"], classes_to_plot=None)})
wandb.log({"ROC Curve" : wandb.plot.roc_curve(test_y, pred_probas, labels=["is_not_flying","is_flying"], classes_to_plot=None)})