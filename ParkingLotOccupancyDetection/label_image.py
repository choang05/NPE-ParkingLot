import cv2
import tensorflow as tf
import sys
from math import floor

# Read the images
image_original = cv2.imread('PL01.jpg')
image_overlay_result = image_original.copy()

image_masks = [cv2.imread('PL01_Slot1_mask.jpg', 0), 
               cv2.imread('PL01_Slot2_mask.jpg', 0), 
               cv2.imread('PL01_Slot3_mask.jpg', 0)]

# Display masked image
#cv2.imshow("Masked Image", image_masked)

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line in tf.gfile.GFile("retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
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

        # Sort to show labels of first prediction in order of confidence
        results = predictions[0].argsort()[-len(predictions[0]):][::-1]
        
        result_top = results[0]

        result_label = label_lines[result_top]
        result_score = predictions[0][result_top]
        #print ('%s (score = %.5f)' % (human_string, score))

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
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontSize = 2
        textThickness = 6
        cv2.putText(image_overlay_result, "Occupancy: " + result_label, (centerX - 350, centerY), font, fontSize, (0, 0, 0), textThickness, cv2.LINE_AA)         
        cv2.putText(image_overlay_result, "Confidence: " + str(result_score*100)[:4 + (1-1)] + '%', (centerX - 350, centerY + textThickness * 10), font, fontSize, (0, 0, 0), textThickness, cv2.LINE_AA)          

        # apply the overlay with transparency, alpha
        alpha = 0.25
        cv2.addWeighted(image_overlay, alpha, image_overlay_result, 1 - alpha, 0, image_overlay_result)

    #   Shrink main image and display
    image_main_small = cv2.resize(image_overlay_result, (0,0), fx=0.20, fy=0.20) 
    cv2.imshow("Result", image_main_small)

#image_main = cv2.resize(image_original, (0,0), fx=0.25, fy=0.25) 
#cv2.imshow("Result", image_main)

while(True):
    # Check end of video
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()