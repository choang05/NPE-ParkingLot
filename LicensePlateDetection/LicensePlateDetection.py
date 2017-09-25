import sys
import os
import threading
import numpy as np
import cv2
import time
import glob
import re
import subprocess
import string
from math import floor
from PIL import Image
from openalpr import Alpr
#
#   Variables
#
#testImagePath = 'ParkingLotOccupancyDetection/PL04.jpg'

# Setup alpr
#alpr = Alpr('us', 'C:/Users/chadh/Documents/GitHub/NPE-ParkingLot/LicensePlateDetection/openalpr.conf', 'C:/Users/chadh/Documents/GitHub/NPE-ParkingLot/LicensePlateDetection/runtime_data')
#if not alpr.is_loaded():
#    print("Error loading OpenALPR")
#    #sys.exit(1)
#alpr.set_top_n(20)
#alpr.set_default_region("tx")

##
##   Functions
##    

def GetLicensePlatePrediction(imagePath):
    print ("Predicting License Plate for " + imagePath + "...")

    #   Call alpr function on image path and save as string
    #subprocess.call(["LicensePlateDetection\\alpr.exe", imagePath], shell = True)
    output = subprocess.check_output(["LicensePlateDetection\\alpr.exe", imagePath], shell = True)

    #   Convert string as list
    outputList = output.decode().split()

    #   check if there was no license plate found. If so, return empty list
    if outputList[0] == "No" and outputList[1] == "license":
        outputList = []
        return outputList

    #   Trim list from unnessary information
    outputList = [x.strip('b') for x in outputList]     #   Remove 'b'
    outputList = [x.strip('-') for x in outputList]     #   Remove '-'
    outputList = [x.strip('confidence:') for x in outputList]     #   Remove 'confidence:'
    outputList = list(filter(None, outputList))         #   Remove empty strings. Must be done AFTER stripping other chars
    outputList.pop(0)   #   Remove 'Plate0'
    outputList.pop(0)   #   Remove # of plates found
    outputList.pop(0)   #   Remove 'results'

    #   Remove confidence values
    del outputList[1::2]
    
    #for string in outputList:
    #    print (string)
    return outputList

#def GetLicensePlateInfo(image):

#    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#    #cv2.imwrite("FrameToExtract.jpg", image)

#    results = alpr.recognize_file('TestData/us-1.jpg')

#    for index, plate in enumerate(results['results']):
#        print("Plate #%d" % index)
#        print("   %12s %12s" % ("Plate", "Confidence"))
#        for candidate in plate['candidates']:
#            prefix = "-"
#            if candidate['matches_template']:
#                prefix = "*"

#            print("  %s %12s%12f" % (prefix, candidate['plate'], candidate['confidence']))
#            #break

#    return results

#    '''
#    for plate in results['results']:
#        i += 1
#        print("Plate #%d" % i)
#        print("   %12s %12s" % ("Plate", "Confidence"))
#        for candidate in plate['candidates']:
#            prefix = "-"
#            if candidate['matches_template']:
#                prefix = "*"

#            print("  %s %12s%12f" % (prefix, candidate['plate'], candidate['confidence']))
#            #break

#    return results
#    '''

##
##   Main
##

# Call when completely done to release memory
#alpr.unload()
#cap.release()
#cv2.destroyAllWindows()
