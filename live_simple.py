from tqdm import tqdm
import cv2
from cv2 import CAP_PROP_FRAME_WIDTH as WIDTH_TAG
from cv2 import CAP_PROP_FRAME_HEIGHT as HEIGHT_TAG
import time

CAM_OVERRIDE = False

cam = cv2.VideoCapture(0)

possible_resolutions = [(16,16),(42,11),(32,32),(40,30),(42,32),(48,32),(60,40),(84,48),(64,64),(72,64),(128,36),(75,64),(150,40),(96,64),(96,64),(128,48),(96,65),(102,64),(101,80),(96,96),(240,64),(160,102),(128,128),(160,120),(160,144),(144,168),(160,152),(160,160),(140,192),(160,200),(224,144),(208,176),(240,160),(220,176),(160,256),(208,208),(256,192),(280,192),(256,212),(432,128),(240,240),(320,192),(320,200),(256,256),(320,208),(320,224),(320,240),(320,256),(376,240),(272,340),(400,240),(320,320),(432,240),(560,192),(400,270),(512,212),(384,288),(480,234),(400,300),(480,250),(312,390),(640,200),(480,272),(512,256),(416,352),(480,320),(640,240),(640,256),(512,342),(400,240),
(512,384),(640,320),(640,350),(640,360),(480,500),(720,348),(720,350),(640,400),(720,364),(800,352),(600,480),(640,480),(640,512),(768,480),(800,480),(848,480),(854,480),(800,600),(960,540),(832,624),(960,544),(1024,576),(960,640),(1024,600),(1024,640),(960,720),(1136,640),(1024,768)]

def get_resolution(cam):
    return cam.get(WIDTH_TAG), cam.get(HEIGHT_TAG)

def set_resolution(cam, width, height):
    cam.set(WIDTH_TAG, width)
    cam.set(HEIGHT_TAG, height)
    set_width, set_height = get_resolution(cam)
    return set_width == width and set_height == height

def get_available_resolutions(cam, possible_resolutions):
    prev_width, prev_height = get_resolution(cam)
    available_resolutions = []
    for width, height in tqdm(possible_resolutions):
        if set_resolution(cam, width, height):
            available_resolutions.append((width,height))
    
    cam.set(WIDTH_TAG, prev_width)
    cam.set(HEIGHT_TAG, prev_height)
    return available_resolutions

if not CAM_OVERRIDE:
    available_resolutions = get_available_resolutions(cam, possible_resolutions)
    target_resolution = (416,416)
    best_resolution = min(available_resolutions, key=lambda x: sum([(a-b)**2 for a,b in zip(x,target_resolution)]))
    set_resolution(cam, *best_resolution)
    print('Selected optimal resolution!')
else:
    set_resolution(cam, *best_resolution)

from utils import draw_boxes
from frontend import YOLO

print('Creating model...', end = ' ')
yolo = YOLO(architecture='Tiny Yolo',
                input_size=416,
                labels=['cube'],
                max_box_per_image=3,
                anchors=[0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828])
yolo.load_weights('save_tiny.h5')
print('Done!')

while True:
    r, frame = cam.read()
    if r:
        start = time.time()
        boxes = yolo.predict(frame, nms_threshold=0.2)
        out_img = draw_boxes(frame, boxes, ['cube'])
        end = time.time()
        print('\rFPS: {:.3f}'.format(1.0/(end-start)), end='')
        cv2.imshow('Image', out_img)
        if cv2.waitKey(1) == ord('q'):
            cam.release()
            cv2.destroyAllWindows()
            break