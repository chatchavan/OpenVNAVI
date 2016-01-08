#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
//////////////////////////////////////////////////////////////////////////////
==============================================================================
OpenVNAVI (Open Vibrotactile Navigation Aid for the Visually Impaired)
Author: David Antón Sánchez
License: GPLv3
github.com/davidanton
==============================================================================
//////////////////////////////////////////////////////////////////////////////
"""

__author__ = "David Anton Sanchez"
__license__ = "GPLv3"


# ============================================================================
# Imports
# ============================================================================
from gevent import monkey
monkey.patch_all()

import cv2
import numpy as np
import sys
import os
import time
import RPi.GPIO as GPIO
from PWM_driver import PWM
from flask import Flask, jsonify, render_template, request, Response
import multiprocessing as mp
import timeit
import ctypes
import gevent
from gevent.wsgi import WSGIServer


# ============================================================================
# Constants
# ============================================================================
""" Each device and channel is identified by an integer. The C++
version of OpenCV defines some constants for the identifiers of
certain devices and channels. However, these constants are not
defined in the Python version and the assignment needs to be
performed manually. """

# Devices.
CV_CAP_OPENNI = 900 # OpenNI (for Microsoft Kinect)
CV_CAP_OPENNI_ASUS = 910 # OpenNI (for Asus Xtion)
# Channels of an OpenNI-compatible depth generator.
CV_CAP_OPENNI_DEPTH_MAP = 0 # Depth values in mm (CV_16UC1)
CV_CAP_OPENNI_POINT_CLOUD_MAP = 1 # XYZ in meters (CV_32FC3)
CV_CAP_OPENNI_DISPARITY_MAP = 2 # Disparity in pixels (CV_8UC1)
CV_CAP_OPENNI_DISPARITY_MAP_32F = 3 # Disparity in pixels (CV_32FC1)
CV_CAP_OPENNI_VALID_DEPTH_MASK = 4 # CV_8UC1
# Channels of an OpenNI-compatible RGB image generator.
CV_CAP_OPENNI_BGR_IMAGE = 5
CV_CAP_OPENNI_GRAY_IMAGE = 6

# ============================================================================
# Global variable
# ============================================================================
isMotorOn = False # Flag for start/stop button.
sourceMode = "kinect"  # Data source "kinect" or "web"


maxPWM = 3000  # TODO: determine proper max
minPWM = 20  # TODO: determine proper min

shared_depthImg16 = None
shared_PWM = None
shared_depthImgFull = None

STREAM_WIDTH = 640 / 4
STREAM_HEIGHT = 480 / 4

# ============================================================================
# Definitions
# ============================================================================
def depthTest():

    """ Grabs frames from different channels, multiplies the output
    by different constant gain values and saves the PNG frames
    for visual inspection of the range of depth values of a scene.
    The folder named "test_frames" must be created. """

    # TODO: possible tuning to change the channel and gain value depending on the environment
    abc = (["a","b","c","d","e","f","g","h","i","j","k","l","m",
          "n","o","p","q","r","s","t","u","v","w","x","y","z"])
    values = 7
    testGain = 5
    maxChannel = 4
    for i in range(0, values):
        for j in range(0, maxChannel):
            capture.grab()
            print "Retrieving channel " + str(j) + abc[i]
            success, rawFrame = capture.retrieve(channel = j)
            frame640j = testGain * i * rawFrame
            cv2.imwrite("test_frames/" + str(j) + abc[i] + ".png", frame640j)

def sweepTest():

    """ Sweeps up and down for all motors. """

    minPWM = 300
    maxPWM = 3000
    step = 100
    for i in range(minPWM,maxPWM,step):
        for row in range(0,8):
            for col in range(0,16):
                IC[row].setPWM(col, 0, i)
                time.sleep(0.001)
                print ("IC: " + str(row) + " motor: " +
                      str(col) + " PWM: " + str(i))

    for i in range(maxPWM,minPWM,-step):
        for row in range(0,8):
            for col in range(0,16):
                IC[row].setPWM(col, 0, i)
                time.sleep(0.001)
                print ("IC: " + str(row) + " motor: " +
                      str(col) + " PWM: " + str(i))

def strobeTest(row):

    """ Strobes maximum and minimum values for a given row. """

    minPWM = 300
    maxPWM = 3000
    IC[row].setAllPWM(0, minPWM)
    time.sleep (1)
    IC[row].setAllPWM(0, maxPWM)
    time.sleep (1)

def fadeIn():

    """ Ramps from low to medium vibration values for a smooth
    transition. """

    minPWM = 300
    maxPWM = 1000
    step = 10
    for i in range(minPWM,maxPWM,step):
        for row in range(0,8):
            IC[row].setAllPWM(0, i)

def fadeOut():

    """ Ramps from medium to low vibration values for a smooth
    transition. """

    minPWM = 300
    maxPWM = 1000
    step = 10
    for i in range(maxPWM, minPWM, -step):
        for row in range(0,8):
            IC[row].setAllPWM(0, i)

def beep(number, t):

    """ Simple buzzer beep. Takes number of beeps and the duration. """

    GPIO.setwarnings(False)
    GPIO.setup(12, GPIO.OUT)
    for i in range (0, number):
        buzzer = GPIO.PWM(12, 5000)
        time.sleep(t)
        buzzer.start(92)
        time.sleep(t)
        buzzer.stop()

def pause():

    """ Pause function. """
    global isMotorOn

    fadeOut()
    for row in range(0, 8):
        IC[row].setAllPWM(0, 0)
    print "System paused by the user."
    time.sleep(0.5)
    beep(1, 0.2)
    print "System ready, press switch to continue..."
    GPIO.wait_for_edge(18, GPIO.RISING)
    fadeIn()
    isMotorOn = True


def depthProceesingPipeline():
    frame = shared_rawFrame[:]
    frame = processDepth(frame)
    frame = scaleFrame(frame)
    return frame

def getFrame():

    """ Gets the frame from the sensor and stores it it.
    Returns 16x8 numpy array. """

    capture.grab()
    success, rawFrame = capture.retrieve(channel = channel)
    
    return rawFrame


def processDepth(inFrame):   
    # crop to match actuator aspect ratio
    frame = inFrame[15:475, 12:618]  # TODO: this aspect ratio doesn't match the vest
    
    # convert to appropriate data type for computation
    frame = frame.astype(float)
    
    # the distance that we are interested
    minDist = 800 # in mm
    maxDist = 2000 # in mm
    
    # save invalid pixels
    invalidIdx = (frame < minDist) | (frame > maxDist)
    
    # scale the value of each pixel
    # reverse the mapping (0: farthest - 255: nearest)
    scaleFactor = 255.0 / (maxDist - minDist)
    frame = 255 - ((frame - minDist) * scaleFactor)
    
    # set invalid pixels to zero
    frame[invalidIdx] = 0
    
    return frame


def scaleFrame(inFrame):
    frame16 = cv2.resize(inFrame, (16,8))
    return frame16


def depthToPWM(frame):
    assert maxPWM > minPWM
    
    # NOTE: Assume that the image processing part already cut to the proper near and far planes
    farDepth = 0
    nearDepth = 255

    scaleFactor = (maxPWM - minPWM) / (nearDepth - farDepth)
    PWM16 = scaleFactor * frame.astype(np.int16) # TODO: allow web to chang the mapping function
    return PWM16


def setVibrationFromPWM(PWM16):

    # cap max PWM to prevent motor from exploding
    PWM16[PWM16 > maxPWM] = maxPWM

    # set the motor
    for row in range(0,8):
        for col in range (0,16):
            IC[row].setPWM(col,0,(PWM16[row,col]))


# ============================================================================
# Vibration renderer process
# ============================================================================

def rendererProcess(webQueue, ipcQueue):

    """ Entry point for the renderer process. """
    global isMotorOn

    # Initialization of PWM drivers. PWM(0x40, debug=True) for debugging.
    global IC
    IC = []
    freq = 490


    # IC.append(PWM(0x40+4))
    # IC.append(PWM(0x40+0))
    # IC.append(PWM(0x40+2))
    # IC.append(PWM(0x40+3))
    # IC.append(PWM(0x40+1))
    # IC.append(PWM(0x40+5))
    # IC.append(PWM(0x40+7))
    # IC.append(PWM(0x40+6))

    # for i in range(0,8):
    #     IC.append(PWM(0x40+i))
    #     IC[i].setPWMFreq(freq)

    # IC[0] = PWM(0x44)
    # IC[7] = PWM(0x46)
    # IC[4] = PWM(0x40)
    # IC[6] = PWM(0x47)

    

    # camera benchmarking
    # getFrame()
    # repCount = 50
    # print "Benchmarking camera for %d frames" % repCount
    # callTime = timeit.timeit("getFrame()", setup="from __main__ import getFrame", number = repCount)
    # print "getFrame() FPS: %.3f" % (repCount / callTime)
    # capture.release()
    # return

    # image processing benchmarking
    # getFrame()
    # repCount = 50
    # print "Benchmarking image processing for %d frames" % repCount
    # callTime = timeit.timeit("depthProceesingPipeline()", setup="from __main__ import depthProceesingPipeline", number = repCount)
    # print "getFrame() FPS: %.3f" % (repCount / callTime)
    # capture.release()
    # return

    

    # GPIO initialization.
    # GPIO.setmode(GPIO.BCM)
    # GPIO.setup(18, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    # sw1 = GPIO.input(18) # Input NO switch.

    # Waits for sw1 to be pressed.
    # print "System ready, press switch to continue..."
    # beep(1, 0.2)
    # GPIO.wait_for_edge(18, GPIO.RISING)
    # fadeIn()
    isMotorOn = True

    shouldTerminate = False

    global sourceMode

    # default transient variables
    sourceMode = "kinect"
    PWMFromWeb = np.ones((8, 16)) * 0

    while not shouldTerminate:

        # check hardware switch
        # gpioValue = GPIO.input(18)
        # if ((gpioValue == True) and (isMotorOn == True)):
        #     pause()

        # process web server message
        if not webQueue.empty():
            requestJson = webQueue.get()

            if "motors" in requestJson:
                PWMFromWeb = np.array(requestJson["motors"])

            if "mode" in requestJson:
                nextMode = requestJson["mode"]
                if nextMode is not sourceMode:
                    print "Switching to mode %s" % nextMode
                    sourceMode = nextMode

        # process inter-process message
        if not ipcQueue.empty():
            ipcCommand = ipcQueue.get()
            if ipcCommand == "terminate":
                shouldTerminate = True

        # prepare PWM
        PWM16 = None
        if sourceMode == "web":
            PWM16 = PWMFromWeb

        elif sourceMode == "kinect":
            frame = depthProceesingPipeline()

            # save a copy of PWM to share with the web server
            global shared_depthImg16
            shared_depthImg16[:] = frame[:]

            # convert to PWM
            PWM16 = depthToPWM(frame)

        # set PWM
        assert PWM16 is not None
        # setVibrationFromPWM(PWM16)

        # save a copy of PWM to share with the web server
        shared_PWM[:] = PWM16[:]


    print "[Renderer] shutdown requested"
    # GPIO.cleanup()
    # for i in range(0, 8):
    #     IC[i].setAllPWM(0, 0)
    print "[Renderer] successfully shutdown"


# ============================================================================
# Camera process
# ============================================================================
def cameraProcess(webQueue, ipcQueue):
    global sensor, channel, capture, shared_rawFrame

    # camera modes
    sensor = CV_CAP_OPENNI_ASUS
    channel = CV_CAP_OPENNI_DEPTH_MAP
    
    # inialize camera
    print "About to initialize camera..."
    capture = cv2.VideoCapture(sensor)
    capture.open(sensor)
    while not capture.isOpened():
        print "Couldn't open sensor. Is it connected? (Trying again in 5 sec)"
        time.sleep(5)
        capture = cv2.VideoCapture(sensor)
        capture.open(sensor)
    print "Sensor opened successfully"

    # get frame and copy to the shared buffer
    shouldTerminate = False
    while not shouldTerminate:
        rawFrame = getFrame()
        shared_rawFrame[:] = rawFrame[:]
        
        # process inter-process message
        # TODO: change to shared ctype value
        if not ipcQueue.empty():
            ipcCommand = ipcQueue.get()
            if ipcCommand == "terminate":
                shouldTerminate = True


    print "[Camera] shutdown requested"
    capture.release()
    print "[Camera] successfully shutdown"


# ============================================================================
# Web server process
# ============================================================================
webServer = Flask(__name__)
webServer.debug = True


@webServer.route("/")
def serve():
    """ Web root. """
    return render_template('index.html')


@webServer.route('/_send_motors', methods=["PUT", "POST"])
def send_motors():

    """ Handles AJAX requests. """

    requestJson = request.get_json()

    # send to renderer process
    webServer.extensions['queue'].put(requestJson)
    return ""


@webServer.route('/_get_render_data')
def get_render_data():
    retVal = {"PWM": shared_PWM.tolist(), "depth": shared_depthImg16.tolist()}
    return jsonify(**retVal)

@webServer.route('/video')
def video():
    return render_template('video.html')

@webServer.route('/video16')
def video16():
    return render_template('video16.html')


def generateDepthImage():
    while True:
        rawFrame = cv2.resize(shared_rawFrame, (STREAM_WIDTH, STREAM_HEIGHT))
        invalidIdx = (rawFrame == 0)

        # convert depth to RGB with proper masking of the invalid pixels
        rawFrame = (rawFrame - 800) / (3500 - 800) * 255  # scale the to gray scale according to the valid depth range
        rawFrameBGR = np.dstack((rawFrame, rawFrame, rawFrame)).view(dtype = np.float64) # expand raw data to BGR dimension
        rawFrameBGR[invalidIdx] = (0, 0, 255)     # mask invalid pixels (too close or no IR feedback) with red
        
        # encode for web streaming
        ret, jpeg = cv2.imencode('.jpg', rawFrameBGR)
        frame = jpeg.tostring()

        # add content header
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        # sleep to work on different thing
        gevent.sleep(0)

def generateDepth16Image():
    while True:
        depth16Frame = shared_depthImg16

        # encode for web streaming
        ret, jpeg = cv2.imencode('.jpg', depth16Frame)
        frame = jpeg.tostring()

        # add content header
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        # sleep to work on different thing
        gevent.sleep(0)


@webServer.route('/feed_depth')
def feed_depth():
    return Response(generateDepthImage(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@webServer.route('/feed_depth16')
def feed_depth16():
    return Response(generateDepth16Image(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@webServer.route('/shutdown')
def shutdown():
    webServer.extensions['cmdQueue'].put("terminate")
    return "Shutting down..."

def webserverProcess(webQueue, ipcQueue):

    """ Entry point for the web server process. """
    if not hasattr(webServer, 'extensions'):
        webServer.extensions = {}
    webServer.extensions['queue'] = webQueue
    webServer.extensions['cmdQueue'] = ipcQueue

    http_server = WSGIServer(('0.0.0.0', 5000), webServer)
    http_server.serve_forever()

    while True:
        ipcCommand = ipcQueue.get()
        if ipcCommand == "terminate":
            print "[WebServer] shutdown requested"
            return


# ============================================================================
# Main function
# ============================================================================
def main():
    global shared_depthImg16, shared_PWM, shared_depthImgFull, shared_rawFrame

    # TODOs:
    print "WARNING: I2C error message is commented out. Chat should uncomment it once David fixes the IC line."

    # for inter-process communication
    webQueue = mp.Queue()
    ipcQueue = mp.Queue()

    # NOTE: ctypes will complain about PEP 3118, but it's fine.
    # (Known ctypes bug: http://stackoverflow.com/questions/4964101/pep-3118-warning-when-using-ctypes-array-as-numpy-array)
    shared_depthImg16_base = mp.Array(ctypes.c_double, 8 * 16)
    shared_depthImg16 = np.ctypeslib.as_array(shared_depthImg16_base.get_obj())
    shared_depthImg16 = shared_depthImg16.reshape(8, 16)

    shared_depthImgFull_base = mp.Array(ctypes.c_double, STREAM_HEIGHT * STREAM_WIDTH)
    shared_depthImgFull = np.ctypeslib.as_array(shared_depthImgFull_base.get_obj())
    shared_depthImgFull = shared_depthImgFull.reshape(STREAM_HEIGHT, STREAM_WIDTH)

    shared_rawFrame_base = mp.Array(ctypes.c_double, 480 * 640)
    shared_rawFrame = np.ctypeslib.as_array(shared_rawFrame_base.get_obj())
    shared_rawFrame = shared_rawFrame.reshape(480, 640)

    shared_PWM_base = mp.Array(ctypes.c_double, 8 * 16)
    shared_PWM = np.ctypeslib.as_array(shared_PWM_base.get_obj())
    shared_PWM = shared_PWM.reshape(8, 16)


    # web server process
    p_webserver = mp.Process(target=webserverProcess, args=(webQueue, ipcQueue))
    p_webserver.start()

    # camera process
    p_camera = mp.Process(target=cameraProcess, args=(webQueue, ipcQueue))
    p_camera.start()

    # jacket renderer process
    p_renderer = mp.Process(target=rendererProcess, args=(webQueue, ipcQueue))
    p_renderer.start()

    workers = [p_webserver, p_renderer, p_camera]
    try:
        for worker in workers:
            worker.join()
    except KeyboardInterrupt:
        print "[Main] received Ctrl + C"
        ipcQueue.put("terminate")
        for worker in workers:
            worker.join()


if __name__ == "__main__":
    main()
