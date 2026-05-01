# usage: python take_webcam_pictures.py --port 1 --storage taken

import numpy as np
import cv2
import os
import argparse
import datetime

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--port", required=True, default=0, type=int, help="port number for a video source") 
ap.add_argument("-s", "--storage", default='storage', help="a directory for captured images") 
args = vars(ap.parse_args())

# assign the port number for a video source and directory name for taken pictures 
port_num = args["port"]
image_root = args["storage"]

# create a storage directory if it does not exist 
now = datetime.datetime.now()

taken_picture_path = image_root
#taken_picture_path = os.path.join(image_root, "{}{:%Y%m%dT%H}".format(image_root, now))

if os.path.exists(taken_picture_path) == -1:
    os.makedirs(taken_picture_path)
    print(taken_picture_path)


# video capture source
cap = cv2.VideoCapture(port_num) # 0, 1, ... depending on # of cameras and the order to a computer  

# in case, two or more images taken in a second,
# provide an unique index number
image_index = 1

# Check success
if not cap.isOpened():
    raise Exception("Could not open video device")

while(True):
    now = datetime.datetime.now()
    #filename = 'capture_' + "{:%Y%m%dT%H%M%S}".format(now) + '.jpg'
    filename = 'capture_' + "{:%Y%m%dT%H%M%S}".format(now) + '_' + str(image_index) + '.jpg'
    imagepath = os.path.sep.join([taken_picture_path, filename])    
        
    ret,frame = cap.read() # return a single frame in variable `frame`
    
    cv2.namedWindow('preview')
    cv2.imshow('preview', frame)
    #cv2.waitKey()

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'): # save on pressing 's' 
        cv2.imwrite(imagepath,frame)
        image_index += 1
        print(imagepath)
    elif key == ord('q'): #quit on pressing 'q' 
        cv2.destroyAllWindows()
        break

cap.release()