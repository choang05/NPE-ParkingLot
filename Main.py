import sys
import os
import threading
import numpy as np
import glob
import re
import cv2
import time
import subprocess
import requests
import json
#import time
#import threading
#from openalpr import Alpr
from multiprocessing import Pool
from LicensePlateDetection.LicensePlateDetection import GetLicensePlatePrediction
from OccupancyDetection.OccupancyDetection import CreateCroppedImages, CreateOverlay, SetupMasks, DeleteOldMasks
from enum import Enum
from math import floor
#from PIL import Image
from difflib import SequenceMatcher
from configparser import SafeConfigParser

#
#   Variables
#
CONFIG_FILE_PATH = 'config.ini'

class Modes(Enum):
    Debug = 0
    Picture = 1
    Video = 2
    Stream = 3

#   Settings
mode = None
frame_capture_delay = None
#video_capture_source = None
images_path = None
results_save_path = None
result_image_size = None
mask_images_path = None
cropped_images_path = None
labels_text_path = None
graph_path = None
post_request_url = None

#   Variables
cameras = []
slots = []
capture = None

##  Image stuff
image_files = []

frameCounter = 0

##
##   Functions
##    
def DeleteResultFiles():
    global results_save_path

    #   Remove any previous image masks in the masks folder
    files = glob.glob(results_save_path + '*')
    for f in files:
        os.remove(f)

def InitializeSlotsData():    
    #   Read json file of license plate data
    with open('data.json') as json_file:  
        json_data = json.load(json_file)
        for camera in json_data['Cameras']:           
            #  Create camera tuple
            camera_tuple = (camera['id'], camera['ipAddress'], camera['slots'])
            # print(camera_tuple[1])
            # for slot in camera_tuple[2]:
            #     for plate in slot['validPlates']:
            #         print (plate['plate'])

            #   Append tuple to array of slots
            cameras.append(camera_tuple)

    #slot_tuple = (camera['id'], camera['ipAddress'], False)

#   initialize the image file paths
def SetupImagesPath(ImagesPath):
    #   Fetch images and convert to array of cv2 image data
    for filename in glob.glob(ImagesPath):
        image_files.append(filename)
    
    #   Sort
    #sort_nicely(image_files)

#def tryint(s):
#    try:
#        return int(s)
#    except:
#        return s

#def alphanum_key(s):
#    """ Turn a string into a list of string and number chunks.
#        "z23a" -> ["z", 23, "a"]
#    """
#    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

#def sort_nicely(l):
#    """ Sort the given list in the way that humans expect.
#    """
#    l.sort(key=alphanum_key)

#   Returns true if the given license plate strings match the one allowed in the parking slot
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

#   Create parking slot validy result
def CreateResults(image):
    global frameCounter
    global result_image_size
    global cropped_images_path

    #   Start timer
    elapsed_time = time.time()

    #   Resize
    #image = cv2.resize(image, (1280, 720), fx=1, fy=1) 

    #   (OccupancyDetection.py)
    CreateCroppedImages(image)

    #cv2.imshow("result", image_result)
    #cv2.imshow("Result", image_result)

    image_masks = []
    license_plates = []

    #   Loop through each mask and adjust validity
    for i, filename in enumerate(glob.glob(cropped_images_path)):
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
    #print(license_plates)

    #   Create overlay
    #image_result = CreateOverlay(image, slots, license_plates)

    #   Shrink image so it takes less space
    ##image_result = cv2.resize(image_result, (result_image_size[0], result_image_size[1]), fx=1, fy=1)
    
    #   Determine if directory exists
    #if not os.path.exists(results_save_path):
    #    print(results_save_path, "doesn't exist! Creating new directory...")
    #    os.makedirs(results_save_path)

    #   save it to file
    #filename = str(frameCounter) + ".jpg"
    #cv2.imwrite(results_save_path + filename, image_result)
    #print(filename, "saved to", results_save_path)

    #   Increase file counter
    frameCounter += 1

    #   Draw image
    #cv2.imshow("frame", image)

    # Check if this is the frame closest to the delay seconds


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

    print("Thread time:", time.time() - elapsed_time)

#   Parse the config file and evaluate
def InitializeConfig():
    print("Initializing config...")

    config = SafeConfigParser()
    config.read(CONFIG_FILE_PATH)

    #   Create references to globals
    global mode
    global frame_capture_delay
    global video_capture_source
    global images_path
    global results_save_path
    global result_image_size
    #global video_stream_address
    global mask_images_path
    global cropped_images_path
    global labels_text_path
    global graph_path
    global post_request_url

    #   Assign variables from config
    #   Main Settings
    mode = Modes(config.getint("Settings", "mode"))
    frame_capture_delay = config.getint("Settings", "frame_capture_delay")
    video_capture_source = config.get("Settings", "video_capture_source")
    images_path = config.get("Settings", "images_path")
    results_save_path = config.get("Settings", "results_save_path")
    result_image_size = [int(i) for i in config.get("Settings", "result_image_size").split(',')] #   List from config returns strings so it needs to be converted to list of ints
    #video_stream_address = config.get("Settings", "video_stream_address")
    mask_images_path = config.get("Settings", "mask_images_path")
    cropped_images_path = config.get("Settings", "cropped_images_path")
    labels_text_path = config.get("Settings", "labels_text_path")
    graph_path = config.get("Settings", "graph_path")
    post_request_url = config.get("Settings", "post_request_url")


##
##   Main
##

def main():  
    #   Refereance globals
    global mode
    global frameCounter
    global frame_capture_delay
    global video_capture_source
    global images_path
    #global video_stream_address
    global mask_images_path
    global labels_text_path
    global graph_path
    global capture
    global post_request_url

    #   Initialize
    InitializeConfig()
    DeleteResultFiles()
    InitializeSlotsData()

    #   Initialize occupancy detection
    DeleteOldMasks()
    SetupMasks(mask_images_path)
    #SetupTensorflow(labels_text_path, graph_path)

    #print ("testing test")
    #startTime = time.time()

    #endTime = time.time()
    #print(str('{0:.3g}'.format(endTime - startTime)) + " Seconds")
    pool = Pool()

    print("Mode:", mode)

    if mode == Modes.Debug:
        print("Debug mode...")

        response = requests.get(post_request_url)
        print(response)

        #json = { "apartment_id": 2, "license_plate": "#87d21b", "_token": "lav9yCUGAx1Ba1lNfI6dHd3VXBJLbjf1cY7WQx0y"}
        response = requests.post(post_request_url, json = { "apartment_id": 2, "license_plate": "#ABC123", "_token": "lav9yCUGAx1Ba1lNfI6dHd3VXBJLbjf1cY7WQx0y"})
        print(response)
        print(response.json())        
    
    ### PICTURE MODE ###
    elif mode == Modes.Picture:
        
        #   Start timer
        elapsed_time = time.time()

        SetupImagesPath(images_path)

        #   for each camera...
        for camera in cameras:

            print('Attempting to connect to camera', camera[0], 'at IP address:', camera[1] + '...')

            #   set capture source of ip address of camera
            capture = cv2.VideoCapture(camera[1])

            # Check if camera opened successfully
            if (capture.isOpened() == True):
                print('Connected!')
                #   Capture frame
                isCaptureSuccess, image = capture.read()

                # Display the resulting frame
                cv2.imshow('frame',image)

                #   if image capture is sucessful... 
                if isCaptureSuccess:
                    #   Create the result from capture image
                    CreateResults(image)

            else:
                print('Could not connect to', camera[1])
                    # print(camera_tuple[1])
            # for slot in camera_tuple[2]:
            #     for plate in slot['validPlates']:
            #         print (plate['plate'])

        print("sleeping for", str(frame_capture_delay), "seconds...")
        time.sleep(frame_capture_delay)
        #for index, imagePath in enumerate(image_files):
        #    image = cv2.imread(imagePath)
        #    thread = threading.Thread(target=CreateResults, args=(image,))
        #    thread.start()

        #image2 = cv2.imread(image_files[1]) 
        #thread2 = threading.Thread(target=CreateResults, args=(image2,))

        #thread1.start()
        #thread2.start()

        #thread1.join()
        #thread2.join()

        print("Thread time:", time.time() - elapsed_time)

    ### VIDEO MODE ###
    #elif mode == Modes.Video:
    #    print("Video mode")

    #    # Create a VideoCapture object
    #    #capture = cv2.VideoCapture(video_capture_source)
    #    capture = None
    #    # Check if camera opened successfully
    #    if (capture.isOpened() == False):
    #        print("Unable to read camera feed")

    #    # Find OpenCV version
    #    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    #    # With webcam get(CV_CAP_PROP_FPS) does not work.
    #    if int(major_ver) < 3:
    #        framerate = capture.get(cv2.cv.CV_CAP_PROP_FPS)
    #        print
    #        "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(framerate)
    #    else:
    #        framerate = capture.get(cv2.CAP_PROP_FPS)
    #        print
    #        "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(framerate)

    #    delayCounter = 0
    #    frameCounter = 0

    #    while(True):

    #        # Capture frame-by-frame
    #        success, image = capture.read()
    #        delayCounter += 1

    #        #   Resize
    #        image_resized = cv2.resize(image, (1280, 720), fx=1, fy=1) 

    #        #   Draw image
    #        cv2.imshow("frame", image_resized)

    #        # Check if this is the frame closest to the delay seconds
    #        if delayCounter == (framerate * frame_capture_delay):
    #            delayCounter = 0
    #            frameCounter += 1

    #            #cv2.imwrite("Frame.jpg", image)

    #            #   Create threads
    #            #thread1 = threading.Thread(target=CaptureFrame, args=())
    #            #thread1 = threading.Thread(target=CreateOverlay, args=())
            
    #            #image_result = Predict(image)
    #            CreateMaskImages(image)
    #            cv2.imwrite("Results/image_result_" + str(frameCounter) + ".jpg", image_result)
            
    #            #thread1.start()
    #            #thread1.join()

    #            #CreateOverlay()

    #        # Check end of video
    #        if cv2.waitKey(25) & 0xFF == ord('q'):
    #            capture.release()
    #            break

    ### STREAM MODE ###
    elif mode == Modes.Stream: 
        #   cache capture source
        #capture = cv2.VideoCapture(video_stream_address)
        capture = None
        # Check if camera opened successfully
        if (capture.isOpened() == True):
            nexttime = time.time()

            while(True):
                #   Capture frame-by-frame
                isCaptureSuccess, image = capture.read()

                #   if image capture is sucessful... 
                if isCaptureSuccess:
                    #   Display capture image
                    #cv2.imshow('frame' + str(frameCounter), image)
                    #   Create the result from capture image
                    CreateResults(image)

                    #   [BUG] if you do not recache the capture after X iterations, it crashes due to an issue with ffmpeg failing to decode stream.
                    if frameCounter % 25 == 0:
                        print("Recaching capture...")
                        #capture = cv2.VideoCapture(video_stream_address)
                        capture = None
                    print("sleeping for", str(frame_capture_delay), "seconds...")
                    time.sleep(frame_capture_delay)

                else:
                    print ("Unable to retrieve image from capture!")

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    capture.release()
                    break
        else:
            print("Unable to read capture source! Trying again in ", str(frame_capture_delay))
            time.sleep(frame_capture_delay)

    # Call when completely done to release memory
    #alpr.unload()
    #sess.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
