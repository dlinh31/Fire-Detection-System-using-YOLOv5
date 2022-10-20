import cv2
import numpy as np

# from coreAI.script import Detect

# detect = Detect()

class VideoCamera(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
    def __del__(self):
        self.cap.release()
    def get_frame(self):
        ret, frame = self.cap.read()
        # frame = detect.dectect(frame)

        frame_flip = cv2.flip(frame, 1)
       
        name, frame = cv2.imencode('.png', frame_flip)
        return frame.tobytes()