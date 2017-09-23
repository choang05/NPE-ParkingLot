import sys
import os
import threading
import numpy as np
import cv2
import time
import glob
import re
import subprocess
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

def GetLicensePlateInfo(imagePath):
    subprocess.call(["LicensePlateDetection\\alpr.exe", imagePath], shell = True)

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
