import cv2
import numpy as np
import time


CV_CAP_OPENNI_ASUS = 910 # OpenNI (for Asus Xtion)

# Channels of an OpenNI-compatible depth generator.
CV_CAP_OPENNI_DEPTH_MAP = 0 # Depth values in mm (CV_16UC1)
CV_CAP_OPENNI_POINT_CLOUD_MAP = 1 # XYZ in meters (CV_32FC3)
CV_CAP_OPENNI_DISPARITY_MAP = 2 # Disparity in pixels (CV_8UC1)
CV_CAP_OPENNI_DISPARITY_MAP_32F = 3 # Disparity in pixels (CV_32FC1)
CV_CAP_OPENNI_VALID_DEPTH_MASK = 4 # CV_8UC1


sensor = CV_CAP_OPENNI_ASUS

capture = cv2.VideoCapture(sensor)
capture.open(sensor)
while not capture.isOpened():
   print "Couldn't open sensor. Is it connected?"
   time.sleep(100)

# capture
dontcare = capture.grab()
success, rawFrame = capture.retrieve(channel = CV_CAP_OPENNI_DEPTH_MAP)

# save single frame
cv2.imwrite("rawFrame.png", rawFrame)



# convert depth to RGB with proper masking of the invalid pixels
outFrame = np.zeros((480, 640, 3), np.uint8)
rawFrameBGR = np.dstack((rawFrame, rawFrame, rawFrame))	# expand raw data to BGR dimension
outFrame[rawFrame == 0] = (0, 0, 255)  # mask invalid pixels (too close or no IR feedback) with red
rawFrameBGRScaled = (rawFrameBGR - 800) * 255 / (3500 - 800)  # scale the to gray scale according to the valid depth range
outFrame[rawFrameBGR != 0] = rawFrameBGRScaled[rawFrameBGR != 0]

cv2.imwrite("outFrame.png", outFrame)

# encode image to byte for streaming
ret, jpeg = cv2.imencode('.jpg', rawFrame)
im_str =  jpeg.tostring()


# BEGIN: video saving (doesn' work)

# video saving
fourcc = cv2.cv.CV_FOURCC('M','J','P','G')
# fourcc = cv2.cv.CV_FOURCC('F', 'M', 'P', '4')
# fourcc = cv2.cv.CV_FOURCC('H','2','6','4')
video = cv2.VideoWriter("video.avi", fourcc, 30, (640, 480))
video.isOpened()  # NOTE: it fails here (15.12.15)


for i in range(50):
	dontcare = capture.grab()
	success, rawFrame = capture.retrieve(channel = CV_CAP_OPENNI_DEPTH_MAP)
	if success:
		video.write(rawFrame)

video.release()

# END: video saving

# cleanup
capture.close()