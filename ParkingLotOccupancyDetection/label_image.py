import cv2
import tensorflow as tf
import sys

# Read the images
image_main = cv2.imread('PL01.jpg')

image_masks = ['PL01_Slot1_mask.jpg', 
                'PL01_Slot2_mask.jpg',
                'PL01_Slot3_mask.jpg']

# Display masked image
#cv2.imshow("Masked Image", image_masked)


## Loads label file, strips off carriage return
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
        mask = cv2.imread(image_masks[index], 0)
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
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))
