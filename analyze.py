import os
import json
import sys

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import utils.nontf_util as nontf_util
import utils.evaluate
from utils.bbox import find_matches
from utils.evaluate import precision, recall


def append_list_dict(ret_dict, dict):
    for key in dict.keys():
        ret_dict[key] += dict[key]
    return ret_dict


def plot_ap(accumulators, average_precission, class_id, text, dir, interpol=True):
    precisions = []
    recalls = []
    for acc in accumulators:
        precisions.append(acc['precision'])
        recalls.append(acc['recall'])
    if interpol:
        precisions = interpolate(precisions)

    fig, ax = plt.subplots()
    plot_pr_curve(ax, class_id, precisions, recalls, average_precission, text)

    outpath = dir + text.replace(" ", "").replace(
        ",", "_")
    if not os.path.exists(dir):
        os.mkdir(dir)
    plt.savefig(outpath)


def plot_pr_curve(ax, class_id, precisions, recalls, average_precision, text):
    ax.step(recalls, precisions, color='b', alpha=0.2,
            where='post')
    ax.fill_between(recalls, precisions, step='post', alpha=0.2,
                    color='b')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('{0:} {1:} : AUC={2:0.2f}'.format(text, utils.evaluate.find_coco_tag(class_id), average_precision))


def interpolate(precisions):
    max_prec = 0
    for i in range(len(precisions) - 1, 0, -1):
        if precisions[i] > max_prec:
            max_prec = precisions[i]
        else:
            precisions[i] = max_prec
    return precisions


def compute_average_precision(accumulators, interpol=True):
    precisions = []
    recalls = []
    for acc in accumulators:
        precisions.append(acc['precision'])
        recalls.append(acc['recall'])

    if interpol:
        precisions = interpolate(precisions)

    # print(precisions)
    # print(recalls)
    previous_recall = 0.
    average_precision = 0.
    for precision, recall in zip(precisions, recalls):
        average_precision += precision * (recall - previous_recall)
        previous_recall = recall
    return average_precision


def accumulate_average_precission(image_data_list, class_id=16, text='', resolution=.01, iou_threshold=0.5,
                                  dir="/Users/saravana/Documents/Work/Projects/Bird_Detection/_bird_detector_output/plots"):
    accumulators = []
    conf_t = 1.01
    while conf_t > 0.:
        utils.nontf_util.print_progress_bar(1000, int(1001 - (conf_t * 1000)))
        acc = {"conf_t": conf_t, "tp": 0, "fp": 0, "fn": 0}
        for i, image_data in enumerate(image_data_list):
            try:
                iou_mask = None
                if len(image_data['output']['bounding_boxes']) > 0:
                    iou_mask = find_matches(image_data, iou_threshold)
                curr_acc = utils.evaluate.evaluate(iou_mask, image_data, conf_t, class_id)
                acc = append_list_dict(acc, curr_acc)
            except:
                pass
        acc['precision'] = utils.evaluate.precision(acc)
        acc['recall'] = utils.evaluate.recall(acc)

        print("conf_t: {:.2}, acc {}".format(conf_t, acc))
        # print(acc)
        accumulators.append(acc)
        conf_t -= resolution
    ap = compute_average_precision(accumulators)
    plot_ap(accumulators, ap, class_id, text, dir)

    if not os.path.exists(dir):
        os.mkdir(dir)
    utils.nontf_util.dump_csv(
        "/Users/saravana/Documents/Work/Projects/Bird_Detection/_bird_detector_output/plots" + text.replace(" ", "").replace(
            ",", "_") + ".csv", accumulators, accumulators[0].keys())


def check_for_class(image_data, class_id, conf_t):
    for bb in image_data['bounding_boxes']:
        if utils.evaluate.find_coco_id(bb['class']) == class_id:
            return True
    for bb in image_data['output']['bounding_boxes']:
        if bb['class_id'] == class_id and bb['score'] >= conf_t:
            return True


def save_image_with_bb(image_path, image_data, trues, save_path, conf_t, classes=[16]):
    if not check_for_class(image_data, 16, conf_t):
        return
    try:
        image = np.array(Image.open(image_path), dtype=np.uint8)
    except:
        print("Problem with path {}".format(image_path))
        return

    # open image_data in matplotlib, remove axis
    fig = plt.figure(frameon=False)
    # fig.set_size_inches(w, h)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image, aspect='auto')

    # correct green, false red
    pred_trues = trues['pred_trues']
    gt_trues = trues['gt_trues']

    for i, gt_bb in enumerate(image_data['bounding_boxes']):
        if utils.evaluate.find_coco_id(gt_bb['class']) not in classes:
            continue
        annotation_text = "gt, {}".format(gt_bb['class'])
        if gt_trues[i]:
            rect = patches.Rectangle((gt_bb['x'], gt_bb['y']), gt_bb['width'], gt_bb['height'], linestyle="dashed",
                                     linewidth=5, edgecolor='b',
                                     facecolor='none')
            ax.add_patch(rect)
            ax.annotate(annotation_text, (gt_bb['x'], gt_bb['y']), color='k', weight='bold',
                        fontsize=16, ha='left', va='bottom', bbox=dict(boxstyle="square", fc="b"))
        else:
            rect = patches.Rectangle((gt_bb['x'], gt_bb['y']), gt_bb['width'], gt_bb['height'], linestyle="dashed",
                                     linewidth=5,
                                     edgecolor='c',
                                     facecolor='none')
            ax.add_patch(rect)
            ax.annotate(annotation_text, (gt_bb['x'], gt_bb['y']), color='k', weight='bold',
                        fontsize=16, ha='left', va='bottom', bbox=dict(boxstyle="square", fc="c"))
    for i, bb in enumerate(image_data['output']['bounding_boxes']):
        if bb['score'] <= conf_t:
            continue
        if bb['class_id'] not in classes:
            continue
        # draw true positives
        annotation_text = "prediction, {}, {:.0%}".format(
            utils.evaluate.find_coco_tag(bb['class_id']), bb['score'])
        if pred_trues[i]:
            rect = patches.Rectangle((bb['x'], bb['y']), bb['width'], bb['height'], linewidth=5,
                                     edgecolor='g',
                                     facecolor='none')
            ax.add_patch(rect)
            ax.add_patch(rect)
            ax.annotate(annotation_text, (bb['x'], bb['y']), color='k', weight='bold',
                        fontsize=16, ha='left', va='bottom', bbox=dict(boxstyle="square", fc="g"))

        else:
            rect = patches.Rectangle((bb['x'], bb['y']), bb['width'], bb['height'], linewidth=5,
                                     edgecolor='r',
                                     facecolor='none')
            ax.add_patch(rect)
            ax.annotate(annotation_text, (bb['x'], bb['y']), color='k', weight='bold',
                        fontsize=16, ha='left', va='bottom', bbox=dict(boxstyle="square", fc="r"))

        # draw true positives

    # plt.show()

    fig.savefig(save_path)
    plt.close(fig)


def test_for_class(image_data, iou_threshold=0.5, conf_t=0.5, save_image=False,
                   image_path=None, save_path=None):
    iou_mask = None
    if len(image_data['output']['bounding_boxes']) > 0:
        iou_mask = find_matches(image_data, iou_threshold)
    res = utils.evaluate.evaluate(iou_mask, image_data, conf_t)

    if save_image:
        trues = utils.evaluate.eval_true_pred(iou_mask, image_data, conf_t)
        save_image_with_bb(image_path, image_data, trues, save_path, conf_t)

    return res


def calc_avg_time(image_data):
    total_time = 0.
    for image in image_data:
        try:
            total_time += image['output']['inf_time']
        except:
            pass
    avg_time = total_time / len(image_data)
    return avg_time


def save_images(paths):
    for thing in paths:
        # prime directories and paths
        abs_path = os.path.join(thing["path"], thing["file"])
        print(abs_path)
        dataset_path = thing["dataset_path"]

        file = open(abs_path)
        data = json.load(file)
        dataset = data['dataset']
        images = dataset['images']

        image_output_dir = os.path.normpath(
            os.path.join(thing['path'], ('images_' + thing['name'])))
        if not os.path.exists(image_output_dir):
            os.mkdir(image_output_dir)
        print("Saving output in {}".format(image_output_dir))

        conf_t = 0.1

        length = len(images)
        for i, image in enumerate(images):
            if not "2011_002144.jpg" in image['path']:
                continue

            nontf_util.print_progress_bar(i, length)

            image_path = os.path.join(dataset_path, image['path'])
            image_out_path = os.path.join(
                image_output_dir, os.path.basename(image['path']))
            try:
                test_for_class(image, iou_threshold=0.50, conf_t=conf_t, save_image=True,
                               image_path=image_path, save_path=image_out_path)
            except Exception as err:
                print("Problem with image_data {}".format(image['path']))
                print(err)


def init_avp(paths):
    for path_obj in paths:
        # prime directories and paths
        abs_path = os.path.join(path_obj["path"], path_obj["file"])

        file = open(abs_path)
        data = json.load(file)
        dataset = data['dataset']
        images = dataset['images']

        accumulate_average_precission(images, 16, path_obj['text'])
        print(path_obj['text'], calc_avg_time(images))


paths = [
    {
        "path": "/Users/saravana/Documents/Work/Projects/Bird_Detection/_bird_detector_output/ssd_resnet50",
        "file": "PascalVOC.json",
        "name": "PascalVOC",
        "dataset_path": "D:/Eigene_Dateien/Documents/GitHub/data/voc_pascal",
        "text": "SSD Resnet50, PascalVOC"
    },
    {
        "path": "/Users/saravana/Documents/Work/Projects/Bird_Detection/_bird_detector_output/rfcn_resnet101",
        "file": "PascalVOC.json",
        "name": "PascalVOC",
        "dataset_path": "D:/Eigene_Dateien/Documents/GitHub/data/voc_pascal",
        "text": "R-FCN Resnet101, PascalVOC"
    },
    {
        "path": "/Users/saravana/Documents/Work/Projects/Bird_Detection/_bird_detector_output/faster_rcnn",
        "file": "PascalVOC.json",
        "name": "PascalVOC",
        "dataset_path": "D:/Eigene_Dateien/Documents/GitHub/data/voc_pascal",
        "text": "Faster R-CNN, PascalVOC"
    },
    # {
    #     "path": "D:/Eigene_Dateien/Documents/GitHub/data/_bird_detector_output/faster_rcnn_2",
    #     "file": "PascalVOC.json",
    #     "name": "PascalVOC",
    #     "dataset_path": "D:/Eigene_Dateien/Documents/GitHub/data/voc_pascal",
    #     "text": "Faster R-CNN 2, PascalVOC"
    # },
    {
        "path": "/Users/saravana/Documents/Work/Projects/Bird_Detection/_bird_detector_output/ssd_resnet50",
        "file": "UCSD_Birds.json",
        "name": "UCSD_Birds",
        "dataset_path": "D:/Eigene_Dateien/Documents/GitHub/data/Caltech-UCSD_Birds",
        "text": "SSD Resnet50, UCSD_Birds"
    },
    {
        "path": "/Users/saravana/Documents/Work/Projects/Bird_Detection/_bird_detector_output/rfcn_resnet101",
        "file": "UCSD_Birds.json",
        "name": "UCSD_Birds",
        "dataset_path": "D:/Eigene_Dateien/Documents/GitHub/data/Caltech-UCSD_Birds",
        "text": "R-FCN Resnet101, UCSD_Birds"
    },
    {
        "path": "/Users/saravana/Documents/Work/Projects/Bird_Detection/_bird_detector_output/faster_rcnn",
        "file": "UCSD_Birds.json",
        "name": "UCSD_Birds",
        "dataset_path": "D:/Eigene_Dateien/Documents/GitHub/data/Caltech-UCSD_Birds",
        "text": "Faster R-CNN, UCSD_Birds"
    },
    # {
    #     "path": "D:/Eigene_Dateien/Documents/GitHub/data/_bird_detector_output/faster_rcnn_2",
    #     "file": "UCSD_Birds.json",
    #     "name": "UCSD_Birds",
    #     "dataset_path": "D:/Eigene_Dateien/Documents/GitHub/data/Caltech-UCSD_Birds",
    #     "text": "Faster R-CNN 2, UCSD_Birds"
    # },
    # {
    #     "path": "D:/Eigene_Dateien/Documents/GitHub/data/_bird_detector_output/ssd_resnet50",
    #     "file": "WindFarmBirds.json",
    #     "name": "WindFarmBirds",
    #     "dataset_path": "D:/Eigene_Dateien/Documents/GitHub/data/wind_farm_birds",
    #     "text": "SSD Resnet50, WindFarmBirds"
    # },
    # {
    #     "path": "D:/Eigene_Dateien/Documents/GitHub/data/_bird_detector_output/rfcn_resnet101",
    #     "file": "WindFarmBirds.json",
    #     "name": "WindFarmBirds",
    #     "dataset_path": "D:/Eigene_Dateien/Documents/GitHub/data/wind_farm_birds",
    #     "text": "R-FCN Resnet101, WindFarmBirds"
    # },
    # {
    #     "path": "D:/Eigene_Dateien/Documents/GitHub/data/_bird_detector_output/faster_rcnn",
    #     "file": "WindFarmBirds.json",
    #     "name": "WindFarmBirds",
    #     "dataset_path": "D:/Eigene_Dateien/Documents/GitHub/data/wind_farm_birds",
    #     "text": "Faster R-CNN, WindFarmBirds"
    # }, {
    #     "path": "D:/Eigene_Dateien/Documents/GitHub/data/_bird_detector_output/faster_rcnn_2",
    #     "file": "WindFarmBirds.json",
    #     "name": "WindFarmBirds",
    #     "dataset_path": "D:/Eigene_Dateien/Documents/GitHub/data/wind_farm_birds",
    #     "text": "Faster R-CNN 2, WindFarmBirds"
    # },

]

if __name__ == "__main__":
    # save_images(paths)
    init_avp(paths)

    #
    # "faster r-cnn"
    # "2011_002144.jpg"
