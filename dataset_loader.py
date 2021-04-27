"""


"""

import xml.etree.ElementTree as ET
import os

class DatasetLoader:
    dataset = None

    print("Start testing for dataset {}.")

    def __init__(self, dataset_path):
        self.dataset = dataset_path


    def __iter__(self):
        pass

def load_pascal_voc(dataset_path,dataset_meta, model):   
    data = {}
    meta = dataset_meta
    meta['progress'] = []
    data['dataset_meta'] = meta
    data['model'] = model

    dataset_obj = {}
    dataset_obj['initialized'] = True
    dataset_obj['classes'] = ["person", "bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train", "bottle", "chair", "dining table", "potted plant", "sofa", "tv/monitor"]
    dataset_obj['data_dirs'] = ["trainval/JPEGImages"]    
    dataset_obj['annotation_dirs'] = ["trainval/Annotations"]

    trainval_jpeg = os.path.join(dataset_path,"trainval/JPEGImages")
    trainval_annotations = os.path.join(dataset_path,"trainval/Annotations")

    images = []
    for i,each in enumerate(os.listdir(trainval_annotations)):
        image_path = os.path.join(trainval_jpeg,os.listdir(trainval_jpeg)[i])
        annotation_path = os.path.join(trainval_annotations,os.listdir(trainval_annotations)[i])

        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        image_obj = {}
        image_obj['path'] = image_path


        for size in root.findall("size"):
            for elem in size:
                if elem.tag == "width":
                    image_obj['width'] = elem.text
                if elem.tag == "height":
                    image_obj['height'] = elem.text

        bounding_boxes = []
        for eachObj in root.findall("object"):
            bounding_box = {}
            for elem in eachObj:
                if elem.tag == "name":
                    bounding_box['class'] = elem.text
                if elem.tag == "bndbox":
                    xmin, ymin, xmax, ymax, bb_width, bb_height = 0, 0, 0, 0, 0, 0
                    for subelem in elem:
                        if subelem.tag == "xmin":
                            xmin = int(subelem.text)
                            bounding_box['x'] = xmin
                        if subelem.tag == "ymin":
                            ymin = int(subelem.text)
                            bounding_box['y'] = ymin
                        if subelem.tag == "xmax":
                            xmax = subelem.text
                        if subelem.tag == "ymax":
                            ymax = subelem.text
                        bb_width = int(xmax) - int(xmin) 
                        bb_height = int(ymax) - int(ymin)
                        bounding_box['width'] = bb_width
                        bounding_box['height'] = bb_height
            bounding_boxes.append(bounding_box)
            image_obj['bounding_boxes'] = bounding_boxes
        
        images.append(image_obj)
        
    dataset_obj['images'] = images
    data['dataset'] = dataset_obj
    return data

