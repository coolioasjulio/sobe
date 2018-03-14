import os
import re

WIDTH = 640
HEIGHT = 360
IMG_FMT = '.png'

with open('dataformat.txt') as f:
    data_fmt = f.read()

with open('objectformat.txt') as f:
    obj_fmt = f.read()

root_folder = 'Boxes'
label_folder = 'Boxes/labels'
annot_folder = os.path.join(root_folder, 'annotations')

if not os.path.isdir(annot_folder):
    os.mkdir(annot_folder)

for label_file in os.listdir(label_folder):
    image_name = label_file[:-4] + IMG_FMT
    img_width = WIDTH
    img_height = HEIGHT
    objects = []
    with open(os.path.join(label_folder, label_file)) as f:
        labels = f.readlines()
        labels = [x.strip() for x in labels if len(x.strip()) > 0]
    for label in labels:
        data = re.split(r'\s', label)
        class_name = 'cube' # data[0]
        x_min, y_min, x_max, y_max = data[4:8]
        objects.append(obj_fmt.format(class_name=class_name, 
                                      x_min=x_min,
                                      y_min=y_min,
                                      x_max=x_max,
                                      y_max=y_max))
    annotation = data_fmt.format(folder_name='boxes-dataset',
                                 image_name=image_name,
                                 img_width=img_width,
                                 img_height=img_height,
                                 objects='\n'.join(objects))
    with open(os.path.join(annot_folder,label_file[:-4] + '.xml'), 'w') as f:
        f.write(annotation)