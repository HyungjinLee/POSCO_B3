#!/usr/bin/env python
# coding: utf-8

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
import pafy 

tf.app.flags.DEFINE_string('model_dir', '',
                           'Model directory (where training data lives)')

tf.app.flags.DEFINE_string('class_type', 'age',
                           'Classification type (age|gender)')


tf.app.flags.DEFINE_string('device_id', '/gpu:0',
                           'What processing unit to execute inference on')

tf.app.flags.DEFINE_string('filename', '',
                           'File (Image) or File list (Text/No header TSV) to process')

tf.app.flags.DEFINE_string('target', '',
                           'CSV file containing the filename processed along with best guess and score')

tf.app.flags.DEFINE_string('checkpoint', 'checkpoint',
                          'Checkpoint basename')

tf.app.flags.DEFINE_string('model_type', 'default',
                           'Type of convnet')

tf.app.flags.DEFINE_string('requested_step', '', 'Within the model directory, a requested step to restore e.g., 9000')

tf.app.flags.DEFINE_boolean('single_look', False, 'single look at the image or multiple crops')

tf.app.flags.DEFINE_string('face_detection_model', '', 'Do frontal face detection with model specified')

tf.app.flags.DEFINE_string('face_detection_type', 'cascade', 'Face detection model type (yolo_tiny|cascade)')

FLAGS = tf.app.flags.FLAGS

RESIZE_FINAL = 227
GENDER_LIST =['M','F']
AGE_LIST = ['10s','20s','30s','40s']
MAX_BATCH_SZ = 32


url = 'https://www.youtube.com/watch?v=MiGqeTPAo3c' 
#url = 'https://www.youtube.com/watch?v=LYCVM9ko0Zo'
video = pafy.new(url)
best=video.getbest(preftype="mp4")

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

def classify_many_single_crop(sess, label_list, softmax_output, coder, images, image, writer):
    try:
        # 속도 개선
        image_batch = make_multi_image_batch(image, coder)
        print('size of batches', image_batch.shape)
        print('session :', sess)
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
        return 'Unidentified', -1

def classify_one_multi_crop(sess, label_list, softmax_output, coder, images, image, writer):
    try:
        # 속도 개선
        image_batch = make_multi_crop_batch(image, coder)
        print('size of batches', image_batch.shape)
        print('session :', sess)
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
        return 'Unidentified', -1


faceProto = "face_model/opencv_face_detector.pbtxt"
faceModel = "face_model/opencv_face_detector_uint8.pb"
eye_cascade = cv2.CascadeClassifier('haarcascae_eye.xml')

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# FACE RECOGNITION MODEL
faceNet = cv2.dnn.readNet(faceModel,faceProto)
coder = ImageCoder()

# Creating different graphs
g1 = tf.Graph()
g2 = tf.Graph()


# AGE MODEL
with g1.as_default() :
    session = tf.Session(graph = g1)
    with session.as_default() :
        images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
        ageNet = select_model('default')
        logits = ageNet(len(AGE_LIST), images, 1, False)
        requested_step = 29999
        checkpoint_path = '%s' % ('/age_model_2')
        model_checkpoint_path, global_step = get_checkpoint(checkpoint_path, requested_step, 'checkpoint')
        age_saver = tf.train.Saver()
        age_saver.restore(session,model_checkpoint_path)

# GENDER MODEL
with g2.as_default() :
    session2 = tf.Session(graph = g2)
    with session2.as_default() :
        images2 = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
        genderNet = select_model('default')
        logits2  = genderNet(len(GENDER_LIST), images2, 1, False)
        requested_step2 = 29999
        checkpoint_path2 = '%s' % ('/gender_model')
        model_checkpoint_path2, global_step = get_checkpoint(checkpoint_path2, requested_step2, 'checkpoint')
        gender_saver = tf.train.Saver()
        gender_saver.restore(session2, model_checkpoint_path2)

softmax_output = tf.nn.softmax(logits)
softmax_output2 = tf.nn.softmax(logits2)

# Open a video file or an image file or a camera stream
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1000)
padding = 20

predict = True # decide if it keeps predicting
blink = False
global a
font = cv2.FONT_HERSHEY_SIMPLEX
diameter=[]
bcount = -1
kernel = np.ones((5,5),np.uint8)

def detect(predict,blink) :
    
    age, gender = 'Unidentified', ''
    
    while 1:
        # Read frame
        t = time.time()
        hasFrame, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray,1.1,7)
        if not hasFrame:
            cv2.waitKey()
            break
        frameFace, bboxes = getFaceBox(faceNet, frame)

        if not bboxes:
            print("No face Detected, Checking next frame")
            predict = True
            #continue    
        
        for bbox in bboxes:
            face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

            #if predict: 
            writer = None
            output = None

            with tf.device('/gpu:0'):
                with g1.as_default():
                    with session.as_default():
                        #age,prob = classify_one_multi_crop(session, AGE_LIST, softmax_output, coder, images, face, writer)
                        age,prob = classify_many_single_crop(session, AGE_LIST, softmax_output, coder, images, face, writer)

            with tf.device('/gpu:1'):
                with g2.as_default():
                    with session2.as_default():
                        #gender,prob = classify_one_multi_crop(session2, GENDER_LIST, softmax_output2, coder, images2, face, writer)
                        gender,prob = classify_many_single_crop(session2, GENDER_LIST, softmax_output2, coder, images2, face, writer)

               # if prob == -1 : # not detected
               #     predict = True
               # else :
               #     predict = False

        label = "{} {}".format(gender,age)
        cv2.putText(frameFace, label, (bbox[0]-5, bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0,255), 2, cv2.LINE_AA)
        cv2.imshow("Age Gender Demo", frameFace)

        print("Time : {:.3f}".format(time.time() - t))
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    
'''
def detect_video(previewName,url) :
    cv2.namedWindow(previewName)
    cap2 = cv2.VideoCapture(url)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH,1000)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT,1000)
    
    if cap2.isOpened():  # try to get the first frame
        rval, frame = cap2.read()   
    i=0
    age, gender = '',''
    #predict = True
    padding = 20
    while 1:
        ret =cap2.grab()
        i=i+1
        frameFace, bboxes = getFaceBox(faceNet, frame)
        ret ,frame =cap2.retrieve()
        cv2.imshow(previewName, frame)
        key = cv2.waitKey(1) & 0xFF
        hasFrame, frame = cap2.read()
        
        if not hasFrame:
            cv2.waitKey()
            break
            
        for bbox in bboxes:
            face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

            #if predict: 
            writer = None
            output = None

            with tf.device('/gpu:0'):
                with g1.as_default():
                    with session.as_default():
                        age,prob = classify_one_multi_crop(session, AGE_LIST, softmax_output, coder, images, face, writer)
            with tf.device('/gpu:1'):
                with g2.as_default():
                    with session2.as_default():
                        gender,prob = classify_one_multi_crop(session2, GENDER_LIST, softmax_output2, coder, images2, face, writer)

             
        label = "{} {}".format(gender,age)
        cv2.putText(frameFace, label, (bbox[0]-5, bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0,255), 2, cv2.LINE_AA)
        cv2.imshow("Age Gender Demo", frameFace)
        
    cv2.destroyWindow(previewName)
'''
def cam(previewName,url):
    cv2.namedWindow(previewName)
    cap2 = cv2.VideoCapture(url)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH,1000)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT,1000)
    
    if cap2.isOpened():  # try to get the first frame
        rval, frame = cap2.read()   
    i=0
    while 1:
        ret =cap2.grab()
        i=i+1
        if i%10==0 :
            ret ,frame =cap2.retrieve()
            cv2.imshow(previewName, frame)
            key = cv2.waitKey(1) & 0xFF
            hasFrame, frame = cap.read()
            if not hasFrame:
                cv2.waitKey()
                break
            frameFace, bboxes = getFaceBox(faceNet, frame)
            if bboxes:
                detect(predict,blink)
            
    cv2.destroyWindow(previewName)

cam("모나리자",best.url)
#detect_video("Detection",best.url)
