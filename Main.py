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
from OccupancyDetection.OccupancyDetection import CreateMaskImages, CreateOverlay, SetupMasks, SetupTensorflow, RemoveOldMasks
from enum import Enum
from math import floor
from PIL import Image
from difflib import SequenceMatcher

#
#   Variables
#

class TestModes(Enum):
    Picture = 1
    Video = 2
    Stream = 3

testmode = TestModes.Picture

frame_capture_delay = 5
video_capture_source = 'OccupancyDetection/PL_Test01_repeat.mp4'
images_path = 'C:/Users/chadh/Desktop/test2Jpegs/*.jpg'
video_stream_address = "http://10.0.0.131:8080/video"

mask_images_path = 'OccupancyDetection/Masks006/*.jpg'

labels_text_path = "OccupancyDetection/retrained_labels.txt"
graph_path = "OccupancyDetection/retrained_graph.pb"

slots = []

##  Image stuff
image_files = []

##
##   Functions
##    
def DeleteResultFiles():
    #   Remove any previous image masks in the masks folder
    files = glob.glob('Results/*')
    for f in files:
        os.remove(f)

def InitializeSlotsData():
    #for i in range (0, 3):
    slot_tuple1 = ('P1', "JLZ8078", False)
    slot_tuple2 = ('P2', "GTF3287", False)
    slot_tuple3 = ('P3', "None", False)
    slot_tuple4 = ('P4', "FTP0928", False)
    slots.append(slot_tuple1)
    slots.append(slot_tuple2)
    slots.append(slot_tuple3)
    slots.append(slot_tuple4)

def SetupImagesPath(ImagesPath):
    #   Fetch images and convert to array of cv2 image data
    for filename in glob.glob(ImagesPath):
        image_files.append(filename)
    
    #   Sort
    sort_nicely(image_files)

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

def GetParkingSlotValidity(slot_tuple, license_plates_to_check):
    ''' Returns a tuple of each parking slot ID and their validity status  '''

    if len(license_plates_to_check) <= 0:
        return False

    #print (slot_tuple[1])
    #print (license_plates_to_check)

    #   loop through plates and check if there is a similarity between the detected plate and the valid plate. If so, return true, else return false
    for index, plate in enumerate(license_plates_to_check):
        if GetStringSimilarity(slot_tuple[1], plate) >= 0.60:
            print("Plate is valid: " + slot_tuple[1] + "(valid plate)" + " and " + plate + "(detected)")
            return True
    print("Plate is NOT valid: " + plate + "(valid plate)")
    return False

#   Given two strings, return a similarity percentage between them in decimal points (0.6, 0.0, 1.0, etc.)
def GetStringSimilarity(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()

##
##   Main
##

DeleteResultFiles()
InitializeSlotsData()

#   Initialize occupancy detection
RemoveOldMasks()
SetupMasks(mask_images_path)
SetupTensorflow(labels_text_path, graph_path)

#print ("testing test")
#startTime = time.time()

#endTime = time.time()
#print(str('{0:.3g}'.format(endTime - startTime)) + " Seconds")

if testmode == TestModes.Picture:
    print("Picture mode")

    SetupImagesPath(images_path)

    for index, imagePath in enumerate(image_files):
        image = cv2.imread(imagePath) 
        #cv2.imwrite('Frame.jpg', image)
    
        #image_result = Predict(image)
        #cv2.imwrite("Results/image_result.jpg", image_result)
        CreateMaskImages(image)

        #cv2.imshow("result", image_result)
        #cv2.imshow("Result", image_result)

        #   Fetch mask images and convert to array of cv2 image data
        mask_images_path = 'OccupancyDetection/MasksCurrent/*.png'
        image_masks = []
        license_plates = []

        for i, filename in enumerate(glob.glob(mask_images_path)):
            image_masks.append(filename)

            #   Get array of license plates predicted for this image mask
            output = GetLicensePlatePrediction(filename)

            license_plates.append(output)

            #   Get the slot validity 
            isValid = GetParkingSlotValidity(slots[i], output)

            #   Update the slot validity
            temp_list = list(slots[i])
            temp_list[2] = isValid
            slots[i] = tuple(temp_list)

        print(slots)

        print(license_plates)

        image_result = CreateOverlay(image, slots, license_plates)
        filename = "Results/" + str(index) + ".jpg"
        cv2.imwrite(filename, image_result)

elif testmode == TestModes.Video:
    print("Video mode")

    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_capture_source)

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
        if delayCounter == (framerate * frame_capture_delay):
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

elif testmode == TestModes.Stream: 
    cap = cv2.VideoCapture(video_stream_address)

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