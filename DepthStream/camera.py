# camera.py
# based on http://www.chioka.in/python-live-video-streaming-example/


import cv2
import numpy as np
import time

CV_CAP_OPENNI_ASUS = 910 # OpenNI (for Asus Xtion)

# Channels of an OpenNI-compatible depth generator.
CV_CAP_OPENNI_DEPTH_MAP = 0 # Depth values in mm (CV_16UC1)
CV_CAP_OPENNI_POINT_CLOUD_MAP = 1 # XYZ in meters (CV_32FC3)
CV_CAP_OPENNI_DISPARITY_MAP = 2 # Disparity in pixels (CV_8UC1): B is x (blue is right), G is y (green is up), and R is z (red is deep)
CV_CAP_OPENNI_DISPARITY_MAP_32F = 3 # Disparity in pixels (CV_32FC1)
CV_CAP_OPENNI_VALID_DEPTH_MASK = 4 # CV_8UC1



class VideoCamera(object):
    def __init__(self):
        sensor = CV_CAP_OPENNI_ASUS
        self.capture = cv2.VideoCapture(sensor)
        self.capture.open(sensor)
        while not self.capture.isOpened():
           print "Couldn't open sensor. Is it connected?"
           time.sleep(100)
    
    def __del__(self):
        self.capture.release()
    
    def get_frame(self):
        dontcare = self.capture.grab()
        success, rawFrame = self.capture.retrieve(channel = CV_CAP_OPENNI_DEPTH_MAP)

        # convert depth to RGB with proper masking of the invalid pixels
        rawFrameBGR = np.dstack((rawFrame, rawFrame, rawFrame)) # expand raw data to BGR dimension
        rawFrameBGRScaled = (rawFrameBGR - 800) * 255 / (3500 - 800)  # scale the to gray scale according to the valid depth range
        outFrame = np.zeros((480, 640, 3), np.uint8)
        outFrame[:] = rawFrameBGRScaled[:]
        outFrame[rawFrame == 0] = (0, 0, 255)    # mask invalid pixels (too close or no IR feedback) with red

        # encode for web streaming
        ret, jpeg = cv2.imencode('.jpg', outFrame)
        return jpeg.tostring()

    def shutdown(self):
        self.capture.release()
        