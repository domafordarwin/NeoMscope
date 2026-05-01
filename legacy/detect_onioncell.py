# detect dividing onion cells in images with mask rcnn model
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from mrcnn.utils import Dataset
import os
import json
from PIL import Image, ImageDraw
import numpy as np
import sys
import time
import skimage
import cv2
import random
from imutils import paths

from config_onion_20201022 import *
from dataset_config import *

# define the prediction configuration
class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "onioncell"
    # number of classes (background + kangaroo)
    NUM_CLASSES = 1 + 5
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    
def load_image(path):
    # Load image
    image = skimage.io.imread(path)
    # If grayscale. Convert to RGB for consistency.
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image)
    # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image



# create config
cfg = PredictionConfig()

# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)

# load model weights
#model_path = 'weights/mask_rcnn_onioncell_1020_0100.h5'
model_path = 'weights/mask_rcnn_onioncell_1020_0089.h5'

model.load_weights(model_path, by_name=True)

# load the COCO class labels our Mask R-CNN was trained on
labelsPath = os.path.sep.join(["object_detection_classes_onion.txt"])
LABELS = open(labelsPath).read().strip().split("\n")
print(LABELS)

# load the set of colors that will be used when visualizing a given
# instance segmentation
colorsPath = os.path.sep.join(["colors.txt"])
COLORS = open(colorsPath).read().strip().split("\n")
COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
COLORS = np.array(COLORS, dtype="uint8")

# cut off value for the confidence 
CONFIDENCE_THRESHOLD = 0.3

output_path = 'detection_results_captured'
#image_root = 'images-train'
#image_root = 'images-test'
image_root = 'captured_raw_images'

#print("[INFO] loading images...")
imagePaths = list(paths.list_images(image_root))
image_files = [pt.split(os.path.sep)[-1] for pt in imagePaths]
#print(image_files)

idx_all = 1
for fname in image_files:
    imagePath = os.path.sep.join([image_root, fname])
    print(imagePath)
    nameonly = fname.split('.')[0]
    
    image = load_image(imagePath)
    (H, W) = image.shape[:2]

    # make prediction
    yhats = model.detect([image], verbose=0)

    for yhat in yhats:
        # clone our original image so we can draw on it
        clone = image.copy()

        print(yhat['rois'])
        print(yhat['class_ids'])
        # plot each box
        for idx, box in enumerate(yhat['rois']):
            #print(idx)
            classIDs = yhat['class_ids']
            #print(classIDs)
            classID = int(classIDs[idx])
            #print(classID)
            confidences = yhat['scores']
            confidence = confidences[idx]
            #print(confidence)

            # get coordinates
            y1, x1, y2, x2 = box
            # calculate width and height of the box
            width, height = x2 - x1, y2 - y1

            # filter out weak predictions by ensuring the detected probability
            # is greater than the minimum probability
            if confidence > CONFIDENCE_THRESHOLD:
                # extract the ROI of the image
                roi = clone[y1:y2, x1:x2]

                color = COLORS[classID-1]

                # draw the bounding box of the instance on the image
                color = [int(c) for c in color]
                cv2.rectangle(clone, (x1, y1), (x2, y2), color, 2)

                text4 = "{}".format(LABELS[classID-1])
                text5 = "{:.4f}".format(confidence)

                cv2.putText(clone, text4, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(clone, text5, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # show the output image
        #pyplot.figure(figsize=(12,12))
        #pyplot.imshow(clone)
        #pyplot.show()

        save_filename = os.path.sep.join([output_path, nameonly + str(idx_all) + '.jpg'])
        cv2.imwrite(save_filename, clone)
        
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
        cv2.moveWindow("output", 100, 50)
        cv2.resizeWindow("output", 400, 400)
        cv2.imshow('output', clone)
        cv2.waitKey(0)

        idx_all += 1


print('DONE!')
