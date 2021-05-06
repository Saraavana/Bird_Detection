import numpy as np
import os
import tensorflow as tf
import pathlib
import time

from matplotlib import pyplot as plt
import matplotlib

from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


import six.moves.urllib as urllib
import sys
import tarfile

import zipfile

from collections import defaultdict
from io import StringIO


from IPython.display import display


def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'

    # base_url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name,
        origin=base_url + model_file,
        untar=True)

    model_dir = pathlib.Path(model_dir) / "saved_model"
    print(model_dir)

    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']

    return model


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    
    
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    
    # Run inference
    output_dict = model(input_tensor)

    
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image_data size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def show_inference(model, image_path):
    # the array based representation of the image_data will be used later in order to prepare the
    # result image_data with boxes and labels on it.
    image_np = np.array(Image.open(image_path))
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)

    matplotlib.use('TkAgg')
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image_np, aspect='auto')
    plt.show(block=True)
    display(Image.fromarray(image_np))


def run_inference(model, image_path):
    image_np = np.array(Image.open(image_path))
    t0 = time.time()
    output_dict = run_inference_for_single_image(model, image_np)
    t1 = time.time()
    output_dict['inf_time'] = t1 - t0
    return output_dict


def output_to_abs(image_dict, output_dict):
    # get classes
    # get absolute values of input
    # multiply, flatten

    w = float(image_dict['width'])
    h = float(image_dict['height'])

    detections = {}
    detections['inf_time'] = output_dict['inf_time']
    detections['num_detections'] = output_dict['num_detections']
    bounding_boxes = []

    for i in range(0, output_dict['num_detections']):
        bounding_box = {}
        bounding_box['class_id'] = int(output_dict['detection_classes'][i])
        # detection['class'] = output_dict['detection_classes']

        detected_bounding_box = output_dict['detection_boxes'][i]
        print("Value 1 is {}".format(detected_bounding_box[1]))
        print("Value 2 is {}".format(w))
        bounding_box['x'] = int(detected_bounding_box[1] * w)
        bounding_box['y'] = int(detected_bounding_box[0] * h)
        bounding_box['width'] = int(detected_bounding_box[3] * w) -bounding_box['x']
        bounding_box['height'] = int(detected_bounding_box[2] * h) - bounding_box['y']

        bounding_box['score'] = float(output_dict['detection_scores'][i])
        bounding_boxes.append(bounding_box)

    detections['bounding_boxes'] = bounding_boxes

    return detections


# Init index on load
# github = os.environ['github']

# PATH_TO_LABELS = os.path.join(github,            'tf_object_detection_api/models/research/object_detection/data/mscoco_label_map.pbtxt')
PATH_TO_LABELS = '/Users/saravana/Documents/Work/Projects/Bird_Detection/models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
