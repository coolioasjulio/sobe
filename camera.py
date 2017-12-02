# Camera processor for the SOBE project
# Made by Jackson Lewis

import threading
import cv2
lock = threading.Lock()
def read_loop(port,cameraprocessor):
    vc = cv2.VideoCapture(port)
    while 1:
        _,f = vc.read()
        lock.acquire()
        cameraprocessor.frame = f
        lock.release()

class CameraProcessor:
    def __init__(self,port):
        self.camport = port
        self.frame = None

    def start_read(self):
        self.process = threading.Thread(target=read_loop,args=(self.camport,self))
        self.process.start()
    def get_latest(self):
        return self.frame
