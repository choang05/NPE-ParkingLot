import sys
import os
import threading
import numpy as np
import cv2
import time
#import tensorflow as tf
import glob
import re
from math import floor
#from PIL import Image
#
#   Variables
#
image_mask_files = []
image_masks = []
label_lines = ""

##
##   Functions
##    

#   Delete any previous image masks in the masks folder
def DeleteOldMasks():
    files = glob.glob('OccupancyDetection/MasksCurrent/*')
    for f in files:
        os.remove(f)

#def SetupTensorflow(labelsTxtFilePath, graphFilePath):
#    # Loads label file, strips off carriage return
#    global label_lines 
#    label_lines = [line.rstrip() for line in tf.gfile.GFile(labelsTxtFilePath)]

#    # Unpersists graph from file
#    f = tf.gfile.FastGFile(graphFilePath, 'rb')
#    graph_def = tf.GraphDef()
#    graph_def.ParseFromString(f.read())
#    _ = tf.import_graph_def(graph_def, name='')

#   Fetch mask images and convert to array of cv2 image data
def SetupMasks(maskImagesPath):
    for filename in glob.glob(maskImagesPath):
        image_mask_files.append(filename)

    sort_nicely(image_mask_files)

    for image in image_mask_files:
        image_masks.append(cv2.imread(image, 0))
    
'''
def Predict(image):
    print ("Starting tensorflow session")
    sess = tf.Session() 

    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    # test image for license plate
    #GetLicensePlateInfo(image)

    #cv2.imshow('image',image)

    #   Create threads
    #thread1 = threading.Thread(target=CaptureFrame, args=())
    #thread1 = threading.Thread(target=CreateOverlay, args=())

    #thread1.start()
    #thread1.join()

    # Read the images
    image_original = image
    #image_original = image
    image_overlay_result = image_original.copy()

    for index, element in enumerate(image_masks):
        #   Create mask
        mask = image_masks[index]
        image_masked = cv2.bitwise_and(image_original, image_original, mask = mask)

        #   Save image to be processed
        #savedImageFile = 'OccupancyDetection/MasksCurrent/image_masked_current_' + str(index) + '.jpg'
        #cv2.imwrite(savedImageFile, image_masked)

        #   Resize
        image_masked_resized = cv2.resize(image_masked, (299, 299), fx=1, fy=1) 
        cv2.imwrite('image_masked_current.jpg', image_masked_resized)

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

        #   Crop to the mask
        x, y, width, height = cv2.boundingRect(contours[0])
        image_mask_cropped = image_masked[y:y+height, x:x+width]
        #   Save image mask
        savedImageFile = 'OccupancyDetection/MasksCurrent/image_masked_current_' + str(index) + '.png'
        cv2.imwrite(savedImageFile, image_mask_cropped)

        #   Create seperate image data for modification
        image_overlay = image_overlay_result.copy()

        #	Determine level of occupancy. Green = empty, Red = occupied, yellow = ???
        #   if the label is occupied and score is atleast %...
        if result_label == label_lines[0] and result_score >= .50:
            cv2.drawContours(image_overlay, contours,-1,(0,0,255), -1)
        #   else if the score is empty and score is atleast 60%...
        elif result_label == label_lines[1] and result_score >= .50:
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
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontSize = .5
        textThickness = 1
        cv2.putText(image_overlay_result, "Occupancy: " + result_label, (centerX, centerY), font, fontSize, (0, 0, 0), textThickness, cv2.LINE_AA)         
        cv2.putText(image_overlay_result, "Confidence: " + str(result_score*100)[:4 + (1-1)] + '%', (centerX, centerY + 20), font, fontSize, (0, 0, 0), textThickness, cv2.LINE_AA)          

        # apply the overlay with transparency, alpha
        alpha = 0.25
        cv2.addWeighted(image_overlay, alpha, image_overlay_result, 1 - alpha, 0, image_overlay_result)

        #   Shrink main image and display
        #image_main_small = cv2.resize(image_overlay_result, (750, 750), fx=1, fy=1) 
        #   Save result image
        #cv2.imwrite('image_overlay_result.jpg', image_main_small)
        #   Display image
        #cv2.imshow("Result", image_main_small)
    
    print ("Closing tensorflow session")
    sess.close()

    return image_overlay_result
'''

#   Given an image, create the masked version from each image mask
def CreateCroppedImages(image):
    print ("Creating Masked Images...")

    #   Adjust mask to correct size
    #image = cv2.resize(image, (1280, 720), fx=1, fy=1)

    # Read the images
    image_original = image
    #image_overlay_result = image

    #   Display image
    cv2.imshow("image", image)

    for i, mask in enumerate(image_masks):
        #   Mask out the image with the mask
        image_masked = cv2.bitwise_and(image_original, image_original, mask = mask)

        #   Save image to be processed
        #savedImageFile = 'OccupancyDetection/MasksCurrent/image_masked_current_' + str(index) + '.jpg'
        #cv2.imwrite(savedImageFile, image_masked)

        #   Draw contour around masks onto main image
        #image_masked_gray = cv2.cvtColor(image_masked,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(image_masks[i], 127,255,0)
        im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        #   Crop to the mask
        x, y, width, height = cv2.boundingRect(contours[0])
        image_masked = image_masked[y:y+height, x:x+width]

        #   Resize mask into a 1:1 aspect ratio (Sqaure)
        #image_masked = cv2.resize(image_masked, (600, 600), fx=1, fy=1)

        #   Save image mask
        savedImageFile = 'OccupancyDetection/MasksCurrent/mask' + str(i) + '.png'
        cv2.imwrite(savedImageFile, image_masked)

        #   Display image
        cv2.imshow("masked image", image_masked)
    
    print ("Masked Images Created.")

#   Returns a overlay image showing validity colors and info
def CreateOverlay(image, slots, licensePlateOutputs):
    print ("Creating Overlay...")

    #   Create threads
    #thread1 = threading.Thread(target=CaptureFrame, args=())
    #thread1 = threading.Thread(target=CreateOverlay, args=())

    #thread1.start()
    #thread1.join()

    # Read the images
    image_original = image
    image_overlay_result = image

    for i, element in enumerate(image_masks):
        #   Create mask
        mask = image_masks[i]

        #   Mask out the image with the mask 
        image_masked = cv2.bitwise_and(image_original, image_original, mask = mask)

        #   Draw contour around masks onto main image
        #image_masked_gray = cv2.cvtColor(image_masked,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(image_masks[i], 127,255,0)
        im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        #   Create seperate image data for modification
        image_overlay = image_overlay_result
        
        #result_score = 0
        slot_id = slots[i][0]
        slot_valid_plate = slots [i][1]
        slot_isValid = slots [i][2]

        #   if no license plates were found...
        #print(licensePlateOutputs)
        if len(licensePlateOutputs[i]) <= 0:
            slot_isValid = "Empty"
            cv2.drawContours(image_overlay, contours,-1,(0, 150, 150), -1)
        #   else if there is a license plate but it is not a valid plate...
        elif slot_isValid == False:
            slot_isValid = "No"
            cv2.drawContours(image_overlay, contours,-1,(0, 0, 255), -1)
        #   else, there is a valid license plate in the right parking slot...
        elif slot_isValid == True:
            slot_isValid = "Yes"
            cv2.drawContours(image_overlay, contours,-1,(0,255,0), -1)

        #   Calculate the moment of the contour moment to calculate center coordinates
        contour_moment = cv2.moments(contours[0])
        centerX = int(contour_moment['m10']/contour_moment['m00'])
        centerY = int(contour_moment['m01']/contour_moment['m00'])

        #   Add adjustments due to text not centered
        centerX = centerX - 200

        #   Text font templates
        #FONT_HERSHEY_SIMPLEX = 0,
        #FONT_HERSHEY_PLAIN = 1,
        #FONT_HERSHEY_DUPLEX = 2,
        #FONT_HERSHEY_COMPLEX = 3,
        #FONT_HERSHEY_TRIPLEX = 4,
        #FONT_HERSHEY_COMPLEX_SMALL = 5,
        #FONT_HERSHEY_SCRIPT_SIMPLEX = 6,
        #FONT_HERSHEY_SCRIPT_COMPLEX = 7,
        #FONT_ITALIC = 16

        #   Text parameters
        verticalSpacing = 50
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontSize = 1
        textThickness = 4
        fontColor = (0,0,0)

        #   Create text
        cv2.putText(image_overlay_result, "Slot: " + str(slot_id), (centerX, centerY), font, fontSize, fontColor, textThickness, cv2.LINE_AA)         
        cv2.putText(image_overlay_result, "Valid Plate: " + slot_valid_plate, (centerX, centerY + verticalSpacing), font, fontSize, fontColor, textThickness, cv2.LINE_AA)         
        cv2.putText(image_overlay_result, "validity: " + str(slot_isValid), (centerX, centerY + verticalSpacing * 2), font, fontSize, fontColor, textThickness, cv2.LINE_AA)            
        #cv2.putText(image_overlay_result, "Confidence: " + str(result_score)[:4 + (1-1)] + '%', (centerX, centerY + verticalSpacing * 3), font, fontSize, (0, 0, 0), textThickness, cv2.LINE_AA)          

        # apply the overlay with transparency, alpha
        alpha = 0.25
        cv2.addWeighted(image_overlay, alpha, image_overlay_result, 1 - alpha, 0, image_overlay_result)

        #   Shrink main image and display
        #image_main_small = cv2.resize(image_overlay_result, (750, 750), fx=1, fy=1) 
        #   Save result image
        #cv2.imwrite('image_overlay_result.jpg', image_main_small)
        #   Display image
        #cv2.imshow("Result", image_main_small)
    
    #print ("Overlay Created")

    return image_overlay_result

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

#
#   Main
#
