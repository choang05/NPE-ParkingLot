import numpy as np
import cv2
import time
import os
from openalpr import Alpr

#
#   Variables
#
frameCaptureDelay = 10
videoCaptureSource = 'TestData/PL_LS_Cropped.mp4'

#
#   Functions
#
def GetLicensePlateInfo(image):

    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imwrite("FrameToExtract.jpeg", image)

    results = alpr.recognize_file('FrameToExtract.jpeg')

    i = 0
    for plate in results['results']:
        i += 1
        print("Plate #%d" % i)
        print("   %12s %12s" % ("Plate", "Confidence"))
        for candidate in plate['candidates']:
            prefix = "-"
            if candidate['matches_template']:
                prefix = "*"

            print("  %s %12s%12f" % (prefix, candidate['plate'], candidate['confidence']))
            break

    return results

#
#   Main
#

# Setup alpr
alpr = Alpr("us", "openalpr.conf", "runtime_data")
if not alpr.is_loaded():
    print("Error loading OpenALPR")
    #sys.exit(1)
alpr.set_top_n(20)
alpr.set_default_region("tx")

# Create a VideoCapture object
cap = cv2.VideoCapture(videoCaptureSource)

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Unable to read camera feed")

# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

# With webcam get(CV_CAP_PROP_FPS) does not work.
# Let's see for ourselves.
if int(major_ver) < 3:
    framerate = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    print
    "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(framerate)
else:
    framerate = cap.get(cv2.CAP_PROP_FPS)
    print
    "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(framerate)

framecount = 0

while(True):
    # Capture frame-by-frame
    success, image = cap.read()
    framecount += 1

    # Check if this is the frame closest to 10 seconds
    if framecount == (framerate * frameCaptureDelay):
        framecount = 0

        # test image for license plate
        GetLicensePlateInfo(image)

        cv2.imshow('image',image)

    # Check end of video
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# Call when completely done to release memory
alpr.unload()