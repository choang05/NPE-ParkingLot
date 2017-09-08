import numpy as np
import cv2
import time
import os
import tensorflow as tf
import sys
import glob
import re
#from openalpr import Alpr
from math import floor
from PIL import Image

#
#   Variables
#
frameCaptureDelay = 1
videoCaptureSource = 'ParkingLotOccupancyDetection/V_03.mp4'

# Setup alpr
#alpr = Alpr("us", "openalpr.conf", "runtime_data")
#if not alpr.is_loaded():
#    print("Error loading OpenALPR")
#    #sys.exit(1)
#alpr.set_top_n(20)
#alpr.set_default_region("tx")
    
# Display masked image
#cv2.imshow("Masked Image", image_masked)

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line in tf.gfile.GFile("ParkingLotOccupancyDetection/retrained_labels.txt")]

##
##   Functions
##

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

##def GetLicensePlateInfo(image):

##    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

##    cv2.imwrite("FrameToExtract.jpg", image)

##    results = alpr.recognize_file('TestData/us-1.jpg')

##    for index, plate in enumerate(results['results']):
##        print("Plate #%d" % index)
##        print("   %12s %12s" % ("Plate", "Confidence"))
##        for candidate in plate['candidates']:
##            prefix = "-"
##            if candidate['matches_template']:
##                prefix = "*"

##            print("  %s %12s%12f" % (prefix, candidate['plate'], candidate['confidence']))
##            #break

##    return results

##    '''
##    for plate in results['results']:
##        i += 1
##        print("Plate #%d" % i)
##        print("   %12s %12s" % ("Plate", "Confidence"))
##        for candidate in plate['candidates']:
##            prefix = "-"
##            if candidate['matches_template']:
##                prefix = "*"

##            print("  %s %12s%12f" % (prefix, candidate['plate'], candidate['confidence']))
##            #break

##    return results
##    '''

##
##   Main
##

#   Fetch mask images and convert to array of cv2 image data
image_mask_files = []
for filename in glob.glob('ParkingLotOccupancyDetection/Masks_03/*.jpg'):
    image_mask_files.append(filename)

sort_nicely(image_mask_files)

image_masks = []
for image in image_mask_files:
    image_masks.append(cv2.imread(image, 0))

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

framecount = 0

# Unpersists graph from file
with tf.gfile.FastGFile("ParkingLotOccupancyDetection/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    while(True):
        # Capture frame-by-frame
        success, image = cap.read()
        framecount += 1

        #   Draw image
        cv2.imshow('image',image)

        # Check if this is the frame closest to the delay seconds
        if framecount == (framerate * frameCaptureDelay):
            framecount = 0

            cv2.imwrite("FrameToExtract.jpg", image)

            # test image for license plate
            #GetLicensePlateInfo(image)

            #cv2.imshow('image',image)

            # Read the images
            image_original = cv2.imread('FrameToExtract.jpg')
            image_overlay_result = image_original.copy()

            for index, element in enumerate(image_masks):
                #mask = cv2.imread(image_masks[index], 0)
                mask = image_masks[index]
                image_masked = cv2.bitwise_and(image_original, image_original, mask = mask)

                #   Save image to be processed
                cv2.imwrite('image_masked_current.jpg', image_masked)

                # Read in the image_data
                image_data = tf.gfile.FastGFile('image_masked_current.jpg', 'rb').read()

                #predictions = sess.run(softmax_tensor, \
                #         {'DecodeJpeg/contents:0': image_data})
                predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

                # Sort for first prediction in order of confidence
                results = predictions[0].argsort()[-len(predictions[0]):][::-1]
        
                result_top = results[0]

                result_label = label_lines[result_top]
                result_score = predictions[0][result_top]
                #print ('Slot %s %s (score = %.5f)' % (index, result_label, result_score))
                #image_main_small = cv2.resize(image_masked, (600, 600), fx=1, fy=1) 
                #cv2.imshow('Slot ' + str(index), image_main_small)

                # for node_id in results:
                #     human_string = label_lines[node_id]
                #     score = predictions[0][node_id]
                #     print('%s (score = %.5f)' % (human_string, score))

                #   Draw contour around masks onto main image
                #image_masked_gray = cv2.cvtColor(image_masked,cv2.COLOR_BGR2GRAY)
                ret,thresh = cv2.threshold(image_masks[index], 127,255,0)
                im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

                #   Create seperate image data for modification
                image_overlay = image_overlay_result.copy()

                #	Determine level of occupancy. Green = empty, Red = occupied, yellow = ???
                #   if the label is occupied and score is atleast 60%...
                if result_label == label_lines[0] and result_score >= .60:
                    cv2.drawContours(image_overlay, contours,-1,(0,0,255), -1)
                #   else if the score is empty and score is atleast 60%...
                elif result_label == label_lines[1] and result_score >= .60:
                    cv2.drawContours(image_overlay, contours,-1,(0,255,0), -1)
                else:
                    cv2.drawContours(image_overlay, contours,-1,(0,150,150), -1)

                #   Calculate the moment of the contour moment to calculate center coordinates
                contour_moment = cv2.moments(contours[0])
                centerX = int(contour_moment['m10']/contour_moment['m00'])
                centerY = int(contour_moment['m01']/contour_moment['m00'])

                #   Draw text
                #FONT_HERSHEY_SIMPLEX = 0,
                #FONT_HERSHEY_PLAIN = 1,
                #FONT_HERSHEY_DUPLEX = 2,
                #FONT_HERSHEY_COMPLEX = 3,
                #FONT_HERSHEY_TRIPLEX = 4,
                #FONT_HERSHEY_COMPLEX_SMALL = 5,
                #FONT_HERSHEY_SCRIPT_SIMPLEX = 6,
                #FONT_HERSHEY_SCRIPT_COMPLEX = 7,
                #FONT_ITALIC = 16
                #font = cv2.FONT_HERSHEY_SIMPLEX
                #fontSize = .5
                #textThickness = 1
                #cv2.putText(image_overlay_result, "Occupancy: " + result_label, (centerX - 350, centerY), font, fontSize, (0, 0, 0), textThickness, cv2.LINE_AA)         
                #cv2.putText(image_overlay_result, "Confidence: " + str(result_score*100)[:4 + (1-1)] + '%', (centerX - 350, centerY + textThickness * 10), font, fontSize, (0, 0, 0), textThickness, cv2.LINE_AA)          

                # apply the overlay with transparency, alpha
                alpha = 0.25
                cv2.addWeighted(image_overlay, alpha, image_overlay_result, 1 - alpha, 0, image_overlay_result)

            #   Shrink main image and display
            image_main_small = cv2.resize(image_overlay_result, (600, 600), fx=1, fy=1) 
            cv2.imshow("Result", image_main_small)

        # Check end of video
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# Call when completely done to release memory
#alpr.unload()
sess.close()
cap.release()
cv2.destroyAllWindows()