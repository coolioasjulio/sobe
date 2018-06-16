#! /usr/bin/env python

"""
This script takes in a configuration file and produces the best model.
The configuration file is a json file and looks like this:
"""

import cli
import os
import numpy as np
from preprocessing import parse_annotation
from frontend import YOLO
import tkinter as tk
from threading import Thread

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(argstate):
    ###############################
    #   Parse the annotations
    ###############################

    # parse annotations of the training set
    train_imgs, train_labels = parse_annotation(argstate.train.annot_folder,
                                                argstate.train.image_folder,
                                                argstate.labels)

    # parse annotations of the validation set, if any, otherwise split the training set
    if os.path.exists(argstate.valid.annot_folder):
        valid_imgs, valid_labels = parse_annotation(argstate.valid.annot_folder,
                                                    argstate.valid.image_folder,
                                                    argstate.labels)
    else:
        train_valid_split = int(0.8*len(train_imgs))
        np.random.seed(42.)
        np.random.shuffle(train_imgs)

        valid_imgs = train_imgs[train_valid_split:]
        train_imgs = train_imgs[:train_valid_split]

    print(train_labels)

    if len(set(argstate.labels).intersection(set(train_labels.keys()))) == 0:
        print("Labels to be detected are not present in the dataset! Please revise the list of labels in the config.json file!")
        return

    ###############################
    #   Construct the model
    ###############################
    global yolo
    yolo = YOLO(architecture=argstate.architecture,
                input_size=argstate.input_size,
                labels=argstate.labels,
                max_box_per_image=argstate.mbpi,
                anchors=argstate.anchors)

    ###############################
    #   Load the pretrained weights (if any)
    ###############################

    if os.path.exists(argstate.pretrained_weights):
        print("Loading pre-trained weights in",
              argstate.pretrained_weights)
        yolo.load_weights(argstate.pretrained_weights)

    # Load pretrained weights into feature extractor and then freeze them
    yolo.feature_extractor.feature_extractor.load_weights('yolov2_weights.h5')
    yolo.feature_extractor.feature_extractor.trainable = True
    
    # Spawn train pauser
    pause_thread = Thread(target=spawn_pause_ui)
    pause_thread.isDaemon = True
    pause_thread.start()
    
    ###############################
    #   Start the training process
    ###############################

    yolo.train(train_imgs=train_imgs,
               valid_imgs=valid_imgs,
               train_times=argstate.train.times,
               valid_times=argstate.valid.times,
               nb_epoch=argstate.nb_epoch,
               learning_rate=argstate.learning_rate,
               batch_size=argstate.batch_size,
               warmup_bs=argstate.warmup_bs,
               object_scale=argstate.object_scale,
               no_object_scale=argstate.no_object_scale,
               coord_scale=argstate.coord_scale,
               class_scale=argstate.class_scale,
               saved_weights_name=argstate.saved_weights_name,
               debug=argstate.debug)

def spawn_pause_ui():
      root = tk.Tk()
      PauseFrame(root).pack(fill='both',expand=True)
      root.mainloop()

class PauseFrame(tk.Frame):
      def __init__(self, parent):
            tk.Frame.__init__(self, parent)
            self.pause_text = tk.StringVar(self, value='Pause')
            self.pause_button = tk.Button(self, textvariable=self.pause_text,
                                          command=self.toggle_pause)
            self.pause_button.pack(fill='both',expand=True)
      
      def toggle_pause(self):
            yolo.pauser.paused = not yolo.pauser.paused
            text = self.pause_text.get()
            self.pause_text.set('Pause' if text=='Resume' else 'Resume')

if __name__ == '__main__':
    argstate = cli.parse_train()
    main(argstate)