import sys
import json
import importlib
import gc

from utils.nontf_util import *
from utils.tf_util import *


# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

os.chdir(os.path.split(__file__)[0])

with open("config.json") as config_file:
    config = json.load(config_file)
    for model_dict in config['object_detectors']:
        if not model_dict['active']:
            continue
        detection_model = load_model(model_dict['name'])
        print("Model {} loaded, start testing.".format(model_dict['abr']))
        for dataset in config['datasets']:
            if not dataset['active']:
                continue
            dataset_path = os.path.normpath(dataset['path'])
            out_dir = os.path.join(os.path.join(
                os.path.normpath(config['output_dir']),
                model_dict['abr']))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            file_name = dataset['short_name'] + ".json"
            out_path = os.path.join(out_dir, file_name)
            sys.path.insert(0, dataset_path)

            cur_dataset_module = importlib.import_module(dataset['loader'])
            cur_dataset = cur_dataset_module.Dataset().getDescriptionDict()

            print("Start testing for dataset {}.".format(dataset['short_name']))

            dataset_length = len(cur_dataset['images'])
            progress, cur_dataset = checkprogress(out_path, cur_dataset)
            for i, image in enumerate(cur_dataset['images']):
                print_progress_bar(i + 1, dataset_length, prefix="{}/{}".format(i + 1, dataset_length))
                if progress > i:
                    continue
                abspath = os.path.join(dataset_path, image['path'])

                try:
                    output = run_inference(detection_model, abspath)
                    image['output'] = output_to_abs(image, output)
                except:
                    print("Problem with image_data {} from dataset {}".format(image['path'], dataset['short_name']))
                dataset['progress'] = [i + 1, dataset_length]
                dump_output(config['output_dir'], model_dict, dataset, cur_dataset)
            del cur_dataset
        tf.keras.backend.clear_session()
        gc.collect()
        del detection_model
print("done")
#
# model_name = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
# detection_model = load_model(model_name)
#
# dataset = windfarmbirdatasetloader.WindFarmBirdDataset()
#
# for image_path in dataset.images:
#     abspath = os.path.join(dataset_path, image_path['path'])
#     show_inference(detection_model, abspath)
#
# del(detection_model)

""" Modelnames

ssd resnet 50:  ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03
faster RCNN:    faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28
RFCN:           rfcn_resnet101_coco_2018_01_28
slowfasterRCNN: faster_rcnn_nas_coco_2018_01_28


"""
