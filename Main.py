import sys
import os
import threading
import numpy as np
import glob
import re
import cv2
import time
import subprocess
#from openalpr import Alpr
from LicensePlateDetection.LicensePlateDetection import GetLicensePlatePrediction
from OccupancyDetection.OccupancyDetection import Predict, CreateMaskImages, CreateOverlay, SetupMasks, SetupTensorflow, RemoveOldMasks
from enum import Enum
from math import floor
from PIL import Image

#
#   Variables
#

class TestModes(Enum):
    Picture = 1
    Video = 2
    Stream = 3

TestMode = TestModes.Picture

frameCaptureDelay = 5
videoCaptureSource = 'OccupancyDetection/PL_Test01_repeat.mp4'
testImagePath = 'OccupancyDetection/PL05.jpg'
videoStreamAddress = "http://10.0.0.131:8080/video"

maskImagesPath = 'OccupancyDetection/Masks005/*.jpg'

labelsTxtFilePath = "OccupancyDetection/retrained_labels.txt"
graphFilePath = "OccupancyDetection/retrained_graph.pb"

##
##   Functions
##    
def DeleteResultFiles():
    #   Remove any previous image masks in the masks folder
    files = glob.glob('Results/*')
    for f in files:
        os.remove(f)

##
##   Main
##

DeleteResultFiles()

#   Initialize occupancy detection
RemoveOldMasks()
SetupMasks(maskImagesPath)
SetupTensorflow(labelsTxtFilePath, graphFilePath)

#print ("testing test")
#startTime = time.time()

#endTime = time.time()
#print(str('{0:.3g}'.format(endTime - startTime)) + " Seconds")

if TestMode == TestModes.Picture:
    print("Picture mode")
    image = cv2.imread(testImagePath) 
    #cv2.imwrite('Frame.jpg', image)
    
    #image_result = Predict(image)
    #cv2.imwrite("Results/image_result.jpg", image_result)
    CreateMaskImages(image)

    #cv2.imshow("result", image_result)
    #cv2.imshow("Result", image_result)

    #   Fetch mask images and convert to array of cv2 image data
    maskImagesPath = 'OccupancyDetection/MasksCurrent/*.png'
    imageMasks = []
    LicensePlates = []
    for filename in glob.glob(maskImagesPath):
        imageMasks.append(filename)
        output = GetLicensePlatePrediction(filename)
        LicensePlates.append(output)
        #print (output)

    image_result = CreateOverlay(image, LicensePlates)
    cv2.imwrite("Results/image_result.jpg", image_result)

elif TestMode == TestModes.Video:
    print("Video mode")

    # Create a VideoCapture object
    cap = cv2.VideoCapture(videoCaptureSource)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Unable to read camera feed")

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    # With webcam get(CV_CAP_PROP_FPS) does not work.
    if int(major_ver) < 3:
        framerate = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        print
        "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(framerate)
    else:
        framerate = cap.get(cv2.CAP_PROP_FPS)
        print
        "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(framerate)

    delayCounter = 0
    frameCounter = 0

    while(True):

        # Capture frame-by-frame
        success, image = cap.read()
        delayCounter += 1

        #   Resize
        image_resized = cv2.resize(image, (1280, 720), fx=1, fy=1) 

        #   Draw image
        cv2.imshow("frame", image_resized)

        # Check if this is the frame closest to the delay seconds
        if delayCounter == (framerate * frameCaptureDelay):
            delayCounter = 0
            frameCounter += 1

            #cv2.imwrite("Frame.jpg", image)

            #   Create threads
            #thread1 = threading.Thread(target=CaptureFrame, args=())
            #thread1 = threading.Thread(target=CreateOverlay, args=())
            
            #image_result = Predict(image)
            CreateMaskImages(image)
            cv2.imwrite("Results/image_result_" + str(frameCounter) + ".jpg", image_result)
            
            #thread1.start()
            #thread1.join()

            #CreateOverlay()

        # Check end of video
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cap.release()
            break

elif TestMode == TestModes.Stream: 
    cap = cv2.VideoCapture(videoStreamAddress)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Unable to read camera feed")

    nexttime = time.time()

    frameCounter = 0

    while(True):

        # Capture frame-by-frame
        success, image = cap.read()

        #   Resize
        image = cv2.resize(image, (1280, 720), fx=1, fy=1) 

        #   Draw image
        #cv2.imshow("frame", image)

        # Check if this is the frame closest to the delay seconds

        frameCounter += 1

        #cv2.imwrite("Frame.jpg", image)

        #   Create threads
        #thread1 = threading.Thread(target=CaptureFrame, args=())
        #thread1 = threading.Thread(target=CreateOverlay, args=())
        #image_result = Predict(image)
        #cv2.imwrite("Results/image_result_" + str(frameCounter) + ".jpg", image_result)
        #cv2.imshow("result", image_result)
        #thread1.start()
        #thread1.join()

        #CreateOverlay()

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cap.release()
            break

# Call when completely done to release memory
#alpr.unload()
#sess.close()
cv2.destroyAllWindows()