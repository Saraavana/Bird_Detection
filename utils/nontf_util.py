import json
import os
import sys
import numpy as np
import xml.etree.ElementTree as ET
import csv
from PIL import Image
import pandas as pd


def image_path_in_dataset(path, dataset):
    for image in dataset['images']:
        if image['path'] == path:
            return True
    return False


def repair_progress_file(outpath, cur_dataset):
    with open(outpath, 'r') as file:
        dataset = json.load(file)
        progress = sys.maxsize
        length = len(cur_dataset['images'])
        print(length)
        for i, image in enumerate(cur_dataset['images']):
            if image_path_in_dataset(image['path'], dataset['dataset']):
                continue
            else:
                if i < progress:
                    print(length)
                    dataset['dataset_meta']['progress'] = [i, length]
                    progress = i
                dataset['dataset']['images'].append(image)
    dump_output("\\Users\\saravana\\Documents\\Work\\Projects\\Bird_Detection\\_bird_detector_output\\", dataset['model'],
                dataset['dataset_meta'], dataset['dataset'])


def checkprogress(outpath, cur_dataset):
    try:
        with open(outpath, 'r') as file:
            dataset = json.load(file)
            return dataset["dataset_meta"]['progress'][0], dataset['dataset']
    except:
        print("No previous status to return to")
        return 0, cur_dataset


def dump_output(path, model_dict, dataset_dict, dataset):
    out_dir = os.path.join(os.path.join(
        os.path.normpath(path),
        model_dict['abr']
    ))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    file_name = dataset_dict['short_name'] + "_tmp" + ".json"
    out_path = os.path.join(out_dir, file_name)

    data = {}
    data['dataset_meta'] = dataset_dict
    data['model'] = model_dict
    data['dataset'] = dataset

    with open(out_path, 'w+') as outfile:
        json.dump(data, outfile)
    try:
        os.remove(out_path.replace("_tmp", ""))
    except:
        pass
    os.rename(out_path, out_path.replace("_tmp", ""))


def dump_csv(path, dict, keys):
    import csv
    try:
        with open(path, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            for data in dict:
                writer.writerow(data)
    except IOError:
        print("I/O error")



def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def bblist_to_nparray(bbs):
    array = np.ndarray(shape=(len(bbs), 4), dtype=int)
    for i, bb in enumerate(bbs):
        array[i] = [bb['x'], bb['y'], bb['x'] + bb['width'], bb['y'] + bb['height']]
    return array

def load_pascal_voc(dataset_path,dataset_meta, model):   
    data = {}
    meta = dataset_meta
    meta['progress'] = []
    data['dataset_meta'] = meta
    data['model'] = model

    dataset_obj = {}
    dataset_obj['initialized'] = True
    dataset_obj['classes'] = ["person", "bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train", "bottle", "chair", "dining table", "potted plant", "sofa", "tv/monitor"]
    
#     dataset_obj['data_dirs'] = ["JPEGImages"]    
#     dataset_obj['annotation_dirs'] = ["Annotations"]


    
    dataset_obj['data_dirs'] = ["trainval/JPEGImages"]    
    dataset_obj['annotation_dirs'] = ["trainval/Annotations"]

    trainval_jpeg = os.path.join(dataset_path,"trainval/JPEGImages")
    trainval_annotations = os.path.join(dataset_path,"trainval/Annotations")
    
#     trainval_jpeg = os.path.join(dataset_path,"JPEGImages")
#     trainval_annotations = os.path.join(dataset_path,"Annotations")

    images = []
    for i,each in enumerate(os.listdir(trainval_annotations)):
        image_path = os.path.join(trainval_jpeg,os.listdir(trainval_jpeg)[i])
        annotation_path = os.path.join(trainval_annotations,os.listdir(trainval_annotations)[i])

        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        image_obj = {}
        image_obj['path'] = image_path
#         image_obj['path'] = "\\Volumes\\My-Passport\\Dataset\\VOCdevkit\\VOC2012\\trainval\\JPEGImages\\2007_000027.jpg"


        for size in root.findall("size"):
            for elem in size:
                if elem.tag == "width":
                    image_obj['width'] = float(elem.text)
                if elem.tag == "height":
                    image_obj['height'] = float(elem.text)

        bounding_boxes = []
        for eachObj in root.findall("object"):
            bounding_box = {}
            for elem in eachObj:
                if elem.tag == "name":
                    bounding_box['class'] = elem.text
                if elem.tag == "bndbox":
                    xmin, ymin, xmax, ymax, bb_width, bb_height = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                    for subelem in elem:
                        if subelem.tag == "xmin":
                            xmin = float(subelem.text)
                            bounding_box['x'] = xmin
                        if subelem.tag == "ymin":
                            ymin = float(subelem.text)
                            bounding_box['y'] = ymin
                        if subelem.tag == "xmax":
                            xmax = subelem.text
                        if subelem.tag == "ymax":
                            ymax = subelem.text
                        bb_width = float(xmax) - float(xmin) 
                        bb_height = float(ymax) - float(ymin)
                        bounding_box['width'] = bb_width
                        bounding_box['height'] = bb_height
            bounding_boxes.append(bounding_box)
            image_obj['bounding_boxes'] = bounding_boxes
        
        images.append(image_obj)
        
    dataset_obj['images'] = images
    data['dataset'] = dataset_obj
    return data


def load_caltech_cub(dataset_path,dataset_meta, model):   
    data = {}
    meta = dataset_meta
    meta['progress'] = []
    data['dataset_meta'] = meta
    data['model'] = model

    images_text_path = os.path.join(dataset_path,"images_sample.txt")

    dataset_obj = {}
    dataset_obj['initialized'] = True
    dataset_obj['classes'] = ["person", "bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train", "bottle", "chair", "dining table", "potted plant", "sofa", "tv/monitor"]

    images = []

    bounding_box_path = os.path.join(dataset_path,"bounding_boxes.txt")
    with open(bounding_box_path) as bb_path:
        bb_lines = bb_path.readlines()
        bb_columns = []

        for bb_line in bb_lines:
            bb_line = bb_line.strip()
            bb_array = [bb_item.strip() for bb_item in bb_line.split(' ')]
            bb_columns.append(bb_array)


    with open(images_text_path) as f:
        lines = f.readlines()
        columns = [] # To store column names

        i = 1
        for line in lines:
            line = line.strip() # remove leading/trailing white spaces
            image_list_array = [item.strip() for item in line.split(' ')]
            columns.append(image_list_array)

        for index,item in enumerate(columns):
            coco_image_path = item[1]
            image_obj = {}
            image_obj['path'] = "images/" + coco_image_path

            
            image = Image.open(os.path.join(dataset_path,image_obj['path']))
            width, height = image.size
            image_obj['width'] = float(width)
            image_obj['height'] = float(width)
            
            bounding_boxes = []
            if item[0] == bb_columns[index][0]:
                bounding_box_obj = {}
                bounding_box_obj['x'] = bb_columns[index][1]
                bounding_box_obj['y'] = bb_columns[index][2]
                bounding_box_obj['width'] = bb_columns[index][3]
                bounding_box_obj['height'] = bb_columns[index][4]
                bounding_boxes.append(bounding_box_obj)

            image_obj['bounding_boxes'] = bounding_boxes
            images.append(image_obj)


    dataset_obj['images'] = images
    data['dataset'] = dataset_obj
    return data

# Filter only bounding boxes of birds class objects only
def filter_bounding_boxes(boxes):
    # Birds-class_id = 15 
    classID = 15
    each_class_id = int(boxes[1])
    if (each_class_id == classID):
        return True
    else:
        return False 

# Write the values to given filename     
def write_to_csv(filename,header=None,data=None):    
    # Open file in append mode
    with open(filename, 'a+', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        csv_df = pd.read_csv(filename, sep = ",") 
        
        headers_from_csv = csv_df.columns.tolist()

        #if there is no existing header in the csv file
        if header != None and not headers_from_csv:
            writer.writerow(header)
        # write multiple rows
        if data != None:
            #check if the data is already present in the csv file
            condition = csv_df["imagepath"]==data[0]
            row = csv_df[condition]      
            items = row.values.tolist()
            # write data to csv, only when there is no row in the csv file with same imagepath
            if not items:
                writer.writerows(data)