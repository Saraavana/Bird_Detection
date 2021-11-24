import numpy as np
import sys
import time

coco_cat = [{"supercategory": "person", "id": 1, "name": "person"},
            {"supercategory": "vehicle", "id": 2, "name": "bicycle"},
            {"supercategory": "vehicle", "id": 3, "name": "car"},
            {"supercategory": "vehicle", "id": 4, "name": "motorcycle", "alt_name": "motorbike"},
            {"supercategory": "vehicle", "id": 5, "name": "airplane", "alt_name": "aeroplane"},
            {"supercategory": "vehicle", "id": 6, "name": "bus"},
            {"supercategory": "vehicle", "id": 7, "name": "train"},
            {"supercategory": "vehicle", "id": 8, "name": "truck"},
            {"supercategory": "vehicle", "id": 9, "name": "boat"},
            {"supercategory": "outdoor", "id": 10, "name": "traffic light"},
            {"supercategory": "outdoor", "id": 11, "name": "fire hydrant"},
            {"supercategory": "outdoor", "id": 13, "name": "stop sign"},
            {"supercategory": "outdoor", "id": 14, "name": "parking meter"},
            {"supercategory": "outdoor", "id": 15, "name": "bench"},
            {"supercategory": "animal", "id": 16, "name": "bird"},
            {"supercategory": "animal", "id": 17, "name": "cat"},
            {"supercategory": "animal", "id": 18, "name": "dog"},
            {"supercategory": "animal", "id": 19, "name": "horse"},
            {"supercategory": "animal", "id": 20, "name": "sheep"},
            {"supercategory": "animal", "id": 21, "name": "cow"},
            {"supercategory": "animal", "id": 22, "name": "elephant"},
            {"supercategory": "animal", "id": 23, "name": "bear"},
            {"supercategory": "animal", "id": 24, "name": "zebra"},
            {"supercategory": "animal", "id": 25, "name": "giraffe"},
            {"supercategory": "accessory", "id": 27, "name": "backpack"},
            {"supercategory": "accessory", "id": 28, "name": "umbrella"},
            {"supercategory": "accessory", "id": 31, "name": "handbag"},
            {"supercategory": "accessory", "id": 32, "name": "tie"},
            {"supercategory": "accessory", "id": 33, "name": "suitcase"},
            {"supercategory": "sports", "id": 34, "name": "frisbee"},
            {"supercategory": "sports", "id": 35, "name": "skis"},
            {"supercategory": "sports", "id": 36, "name": "snowboard"},
            {"supercategory": "sports", "id": 37, "name": "sports ball"},
            {"supercategory": "sports", "id": 38, "name": "kite"},
            {"supercategory": "sports", "id": 39, "name": "baseball bat"},
            {"supercategory": "sports", "id": 40, "name": "baseball glove"},
            {"supercategory": "sports", "id": 41, "name": "skateboard"},
            {"supercategory": "sports", "id": 42, "name": "surfboard"},
            {"supercategory": "sports", "id": 43, "name": "tennis racket"},
            {"supercategory": "kitchen", "id": 44, "name": "bottle"},
            {"supercategory": "kitchen", "id": 46, "name": "wine glass"},
            {"supercategory": "kitchen", "id": 47, "name": "cup"},
            {"supercategory": "kitchen", "id": 48, "name": "fork"},
            {"supercategory": "kitchen", "id": 49, "name": "knife"},
            {"supercategory": "kitchen", "id": 50, "name": "spoon"},
            {"supercategory": "kitchen", "id": 51, "name": "bowl"},
            {"supercategory": "food", "id": 52, "name": "banana"},
            {"supercategory": "food", "id": 53, "name": "apple"},
            {"supercategory": "food", "id": 54, "name": "sandwich"},
            {"supercategory": "food", "id": 55, "name": "orange"},
            {"supercategory": "food", "id": 56, "name": "broccoli"},
            {"supercategory": "food", "id": 57, "name": "carrot"},
            {"supercategory": "food", "id": 58, "name": "hot dog"},
            {"supercategory": "food", "id": 59, "name": "pizza"},
            {"supercategory": "food", "id": 60, "name": "donut"},
            {"supercategory": "food", "id": 61, "name": "cake"},
            {"supercategory": "furniture", "id": 62, "name": "chair"},
            {"supercategory": "furniture", "id": 63, "name": "couch", "alt_name": "sofa"},
            {"supercategory": "furniture", "id": 64, "name": "potted plant", "alt_name": "pottedplant"},
            {"supercategory": "furniture", "id": 65, "name": "bed"},
            {"supercategory": "furniture", "id": 67, "name": "dining table", "alt_name": "diningtable"},
            {"supercategory": "furniture", "id": 70, "name": "toilet"},
            {"supercategory": "electronic", "id": 72, "name": "tv", "alt_name": "tvmonitor"},
            {"supercategory": "electronic", "id": 73, "name": "laptop"},
            {"supercategory": "electronic", "id": 74, "name": "mouse"},
            {"supercategory": "electronic", "id": 75, "name": "remote"},
            {"supercategory": "electronic", "id": 76, "name": "keyboard"},
            {"supercategory": "electronic", "id": 77, "name": "cell phone"},
            {"supercategory": "appliance", "id": 78, "name": "microwave"},
            {"supercategory": "appliance", "id": 79, "name": "oven"},
            {"supercategory": "appliance", "id": 80, "name": "toaster"},
            {"supercategory": "appliance", "id": 81, "name": "sink"},
            {"supercategory": "appliance", "id": 82, "name": "refrigerator"},
            {"supercategory": "indoor", "id": 84, "name": "book"},
            {"supercategory": "indoor", "id": 85, "name": "clock"},
            {"supercategory": "indoor", "id": 86, "name": "vase"},
            {"supercategory": "indoor", "id": 87, "name": "scissors"},
            {"supercategory": "indoor", "id": 88, "name": "teddy bear"},
            {"supercategory": "indoor", "id": 89, "name": "hair drier"},
            {"supercategory": "indoor", "id": 90, "name": "toothbrush"}]


# Helper functions
def find_coco_id(tag):
    for cat in coco_cat:
        if cat['name'] == tag:
            return cat['id']
        if "alt_name" in cat.keys():
            if cat['alt_name'] == tag:
                return cat['id']
    # print("could not find id for {}".format(tag))
    return 0


def find_coco_tag(id):
    for cat in coco_cat:
        if cat['id'] == id:
            return cat['name']
    return "id {} is unknown".format(id)


def get_pred_classes(image):
    pred = image['output']['bounding_boxes']
    pred_classes = []
    for bb in pred:
        pred_classes.append(bb['class_id'])
    return np.array(pred_classes)


def get_gt_classes(image):
    pred = image['bounding_boxes']
    pred_classes = []
    for bb in pred:
        pred_classes.append(find_coco_id(bb['class']))
    return np.array(pred_classes)


def get_pred_conf(image):
    pred = image['output']['bounding_boxes']
    pred_conf = []
    for bb in pred:
        pred_conf.append(bb['score'])
    return np.array(pred_conf)


# precission and recall
def precision(dict):
    total_predicted = dict['tp'] + dict['fp']
    if total_predicted == 0:
        total_gt = dict['tp'] + dict['fn']
        if total_gt == 0:
            return 1.
        else:
            return 0.
    return dict['tp'] / total_predicted


def recall(dict):
    total_gt = dict['tp'] + dict['fn']
    if total_gt == 0:
        return 1.
    return dict['tp'] / total_gt


# detemining tp, fp, fn
def evaluate(IoUmask, image, confidence_threshold, class_id=16):
    # accumulators: number of classes ... we are not interested
    # only birds are interesting, so our i is 16
    acc = {'fp': 0, 'tp': 0, 'fn': 0}
    gt_classes = get_gt_classes(image)
    pred_classes = get_pred_classes(image)
    pred_conf = get_pred_conf(image)

    # we are only interested in birds
    # birds are the class number 16, everything else isn't interesting
    gt_number = np.sum(gt_classes == class_id)
    # print(pred_conf)
    # time.wait(1)

    pred_mask = np.logical_and(pred_classes == class_id, pred_conf >= confidence_threshold)
    pred_number = np.sum(pred_mask)
    if pred_number == 0:
        acc['fn'] += gt_number
        return acc

    IoU1 = IoUmask[pred_mask, :]
    mask = IoU1[:, gt_classes == class_id]


    tp = np.sum(mask.any(axis=0))
    fp = pred_number - tp
    fn = gt_number - tp

    acc['tp'] += tp
    acc['fn'] += fn
    acc['fp'] += fp
    return acc


def eval_true_pred(IoUmask, image, confidence_threshold):
    gt_classes = get_gt_classes(image)
    pred_classes = get_pred_classes(image)
    pred_conf = get_pred_conf(image)

    pred_trues = np.zeros((pred_classes.shape), dtype=bool)
    gt_trues = np.zeros((gt_classes.shape), dtype=bool)

    if IoUmask is None:
        return {"pred_trues": pred_trues, "gt_trues": gt_trues}
    # we are only interested in birds
    # birds are the class number 16, everything else isn't interesting
    all_res = np.zeros((IoUmask.shape), dtype=bool)
    for i in range(1, len(coco_cat) + 1):
        gt_mask = gt_classes == i

        pred_mask = np.logical_and(pred_classes == i, pred_conf >= confidence_threshold)

        IoU1 = np.copy(IoUmask)
        for k in range(len(gt_mask)):
            IoU1[:, k] = np.logical_and(IoU1[:, k], pred_mask)
        for j in range(len(pred_mask)):
            IoU1[j, :] = np.logical_and(IoU1[j, :], gt_mask)
        all_res = np.logical_or(all_res, IoU1)

    for i in range(len(pred_classes)):
        for j in range(len(gt_classes)):
            if all_res[i, j]:
                pred_trues[i] = True
                gt_trues[j] = True

    # print(pred_classes, pred_trues, " | ", gt_classes, gt_trues)

    return {"pred_trues": pred_trues, "gt_trues": gt_trues}
