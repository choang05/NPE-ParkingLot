import cv2
import tensorflow as tf
import sys

# Read the images
image_main = cv2.imread('PL01.jpg')
image_overlay = image_main.copy()

image_masks = [cv2.imread('PL01_Slot1_mask.jpg', 0), 
               cv2.imread('PL01_Slot2_mask.jpg', 0), 
               cv2.imread('PL01_Slot3_mask.jpg', 0)]

# Display masked image
#cv2.imshow("Masked Image", image_masked)

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("retrained_labels.txt")]

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
        image_masked = cv2.bitwise_and(image_main, image_main, mask = mask)

        cv2.imwrite('image_masked.jpg', image_masked)

        # change this as you see fit
        image_path = 'image_masked.jpg'

        # Read in the image_data
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()

        #predictions = sess.run(softmax_tensor, \
        #         {'DecodeJpeg/contents:0': image_data})
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        results = predictions[0].argsort()[-len(predictions[0]):][::-1]
        
        result_top = results[0]

        human_string = label_lines[result_top]
        score = predictions[0][result_top]
        print('%s (score = %.5f)' % (human_string, score))

        # for node_id in results:
        #     human_string = label_lines[node_id]
        #     score = predictions[0][node_id]
        #     print('%s (score = %.5f)' % (human_string, score))

        #   Draw contour around masks onto main image
        #image_masked_gray = cv2.cvtColor(image_masked,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(image_masks[index], 127,255,0)
        im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        #	Determine color of occupancy
        if human_string == label_lines[0]:
			cv2.drawContours(image_main, contours, -1, (255,0,0), -1)
    	else:
			cv2.drawContours(image_main, contours, -1, (0,255,0), -1)

#   Shrink main image and display
image_main_small = cv2.resize(image_main, (0,0), fx=0.25, fy=0.25) 
cv2.imshow("Result", image_main_small)


while(True):
    # Check end of video
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()