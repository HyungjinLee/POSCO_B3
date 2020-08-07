from utils import *
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import tensorflow as tf
from model import select_model, get_checkpoint

RESIZE_FINAL = 227
GENDER_LIST =['M','F']
AGE_LIST = ['(18,19)','(20,24)','(25,29)','(30,34)','(35,39)']
MAX_BATCH_SZ = 128

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

def classify_one_multi_crop(sess, label_list, softmax_output, coder, images, image, writer):
    try:

        image_batch = make_multi_crop_batch(image, coder)
        print('size of batches', image_batch.shape)
        
        batch_results = sess.run(softmax_output, feed_dict={images:image_batch.eval(session=sess)})
        output = batch_results[0]
        batch_sz = batch_results.shape[0]
        for i in range(1, batch_sz):
            output = output + batch_results[i]
        
        output /= batch_sz
        best = np.argmax(output)
        best_choice = (label_list[best], output[best])
        print('Guess @ 1 %s, prob = %.2f' % best_choice)
    
        if writer is not None:
            writer.writerow((best_choice[0], '%.2f' % best_choice[1]))
        return best_choice
    
    except Exception as e:
        print(e)
        print('Failed to run image')
        return '인식', '안됨'
    
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# FACE RECOGNITION MODEL
faceNet = cv2.dnn.readNet(faceModel,faceProto)

config = tf.ConfigProto(allow_soft_placement=True)

sess = tf.Session()    
images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])

# AGE MODEL
ageNet = select_model('default')
logits = ageNet(len(AGE_LIST), images, 1, False)
images2 = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])

requested_step = 14999

'''
# GENDER MODEL
genderNet = select_model('bn')

logits2 = genderNet(len(GENDER_LIST), images2, 1, False)

#init = tf.global_variables_initializer()
#requested_step = FLAGS.requested_step if FLAGS.requested_step else None

requested_step2 = 29999

#checkpoint_path = '%s' % (FLAGS.model_dir)
'''

checkpoint_path = '%s' % ('C:/Users/LEE/Desktop/rude-carnie/age_model')
#checkpoint_path2 = '%s' % ('C:/Users/LEE/Desktop/rude-carnie/gender_model')

model_checkpoint_path, global_step = get_checkpoint(checkpoint_path, requested_step, 'checkpoint')
#model_checkpoint_path2, global_step = get_checkpoint(checkpoint_path2, requested_step2, 'checkpoint')

saver = tf.train.Saver()
#saver2 = tf.train.Saver()
saver.restore(sess, model_checkpoint_path)
#saver2.restore(sess2, model_checkpoint_path2)
coder = ImageCoder()

# Open a video file or an image file or a camera stream
cap = cv2.VideoCapture(1)
padding = 20

while 1:
    # Read frame
    t = time.time()
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv2.waitKey()
        break
    frameFace, bboxes = getFaceBox(faceNet, frame)
    
    if not bboxes:
        print("No face Detected, Checking next frame")
        continue

    for bbox in bboxes:
        # print(bbox)
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
        with tf.device('/cpu:0'):
            #softmax_output = tf.nn.softmax(logits)
            #softmax_output2 = tf.nn.softmax(logits2)

            writer = None
            output = None
        
            #age,prob = classify_one_multi_crop(sess, AGE_LIST, softmax_output, coder, images, face, writer)
            #gender,prob = age,prob = classify_one_multi_crop(sess, GENDER_LIST, softmax_output2, coder, images2, face, writer)
             
        label = "{}".format('123')
        cv2.putText(frameFace, label, (bbox[0]-5, bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0,255), 2, cv2.LINE_AA)
        cv2.imshow("Age Gender Demo", frameFace)
    
    print("Time : {:.3f}".format(time.time() - t))
    key = cv2.waitKey(1) & 0xFF
    
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()