# detect kangaroos in photos with mask rcnn model
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
import datetime
import skimage
import cv2
import random

import imutils
from imutils import paths
from imutils.video import VideoStream

from config_onion_20201022 import *
from dataset_config import *
import tensorflow as tf
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



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
CONFIDENCE_THRESHOLD = 0.7
conf_threshold = CONFIDENCE_THRESHOLD

output_path = 'detection_results_live'
uploadPath = 'captured_raw_images'

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
print('Running inference for USB camera')

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # Start timer (for calculating frame rate)
    current_count=0
    t1 = cv2.getTickCount()
    
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 600 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=600)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]

    # show the output frame and wait for a key press
    key = cv2.waitKey(1) & 0xFF

    # make prediction
    yhats = model.detect([frame], verbose=0)

    for yhat in yhats:
        # clone our original image so we can draw on it
        #clone = image.copy()

        #print(yhat['rois'])
        #print(yhat['class_ids'])

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

            text4 = "{}".format(LABELS[classID-1])
            text5 = "{:.4f}".format(confidence)
            #print(text4)
            #print(text5)

            # get coordinates
            y1, x1, y2, x2 = box
            # calculate width and height of the box
            width, height = x2 - x1, y2 - y1

            # filter out weak predictions by ensuring the detected probability
            # is greater than the minimum probability
            if confidence > conf_threshold:
                # extract the ROI of the image
                roi = frame[y1:y2, x1:x2]

                color = COLORS[classID-1]

                # draw the bounding box of the instance on the image
                color = [int(c) for c in color]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                text4 = "{}".format(LABELS[classID-1])
                text5 = "{:.4f}".format(confidence)

                cv2.putText(frame, text4, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, text5, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                current_count+=1

        # Draw framerate in corner of frame
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(15,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,55),2,cv2.LINE_AA)
        cv2.putText (frame,'Total Detection Count : ' + str(current_count),(15,65),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,55),2,cv2.LINE_AA)
        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object Detector', frame)

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1


        # if the `q` key was pressed, break from the loop
        if key == ord("s"):
            ts = datetime.datetime.now()
            filename_header = '{}'.format(ts.strftime('%Y-%m-%d_%H-%M-%S'))
            imagepath = os.path.sep.join([uploadPath, filename_header + '.jpg'])
            save_filename = os.path.sep.join([output_path, filename_header + '-detected.jpg'])
            cv2.imwrite(save_filename, frame)
            

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

