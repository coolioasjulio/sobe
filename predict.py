#! /usr/bin/env python

import cli
import os
import cv2
import numpy as np
from tqdm import tqdm
from utils import draw_boxes
from frontend import YOLO
import time
import sys
from utils import BoundBox

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(argstate):

    weights_path = argstate.weights
    image_path = argstate.input

    ###############################
    #   Make the model
    ###############################
    global yolo
    try:
        yolo.anchors
    except:
          yolo = YOLO(architecture=argstate.architecture,
                      input_size=argstate.input_size,
                      labels=argstate.labels,
                      max_box_per_image=argstate.max_box_per_image,
                      anchors=argstate.anchors)
      
          ###############################
          #   Load trained weights
          ###############################
      
          yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes
    ###############################

    # if it's an image, do detection, save image with bounding boxes to the same folder

    # if it's a folder, do detection, save images with boundins boxes to another folder

    # if result folder is present, save annotations to the result folder
    global image
    if image_path[-4:] == '.mp4':
        video_out = image_path[:-4] + '_detected' + image_path[-4:]

        video_reader = cv2.VideoCapture(image_path)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_writer = cv2.VideoWriter(video_out,
                                       cv2.VideoWriter_fourcc(*'MPEG'),
                                       20.0,
                                       (frame_w, frame_h))

        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()

            boxes = yolo.predict(image, nms_threshold=0.3)
            image = draw_boxes(image, boxes, argstate.labels)

            video_writer.write(np.uint8(image))

        video_reader.release()
        video_writer.release()
    else:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (416,416))
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        start = time.time()
        boxes = yolo.predict(img_rgb)
        end = time.time()
        print('Prediction took {} seconds!'.format(end-start))
        print(len(boxes), 'boxes are found')
        try:
              boxes.append(get_ground_truth(image_path))
        except:
              pass
        image = draw_boxes(image, boxes, argstate.labels)


        # cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)
        
        cv2.imshow('image', image)
        cv2.waitKey(0)

def get_ground_truth(image_path):
      annot_name = image_path[image_path.rfind('\\')+1:image_path.rfind('.')] + '.xml'
      annot_name = os.path.join('robot-dataset','annotations',annot_name)
      with open(annot_name) as f:
            annot = f.read()
            global xmin,xmax,ymin,ymax
            xmin = float(annot[annot.find('<xmin>')+6:annot.find('</xmin>')])
            xmax = float(annot[annot.find('<xmax>')+6:annot.find('</xmax>')])
            ymin = float(annot[annot.find('<ymin>')+6:annot.find('</ymin>')])
            ymax = float(annot[annot.find('<ymax>')+6:annot.find('</ymax>')])
            x = np.average((xmin,xmax))/416
            y = np.average((ymin,ymax))/416
            w = (xmax-xmin)/416
            h = (ymax-ymin)/416
            box = BoundBox(x,y,w,h,1.0,np.array((0.0,1.0)))
            return box

if __name__ == '__main__':
    while True:
          if(len(sys.argv) > 1):
              argstate = cli.parse_predict()
          else:
              class A():
                  pass
              argstate = A()
              argstate.architecture = 'Full Yolo'
              argstate.input_size = 416
              argstate.labels = ['robot','true_robot']
              argstate.max_box_per_image = 3
              argstate.anchors = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
              argstate.weights = 'robot_full_yolo_pretrained.h5'
              
              argstate.input = os.path.join('robot-dataset','images',np.random.choice(os.listdir('robot-dataset/images')))
          main(argstate)
