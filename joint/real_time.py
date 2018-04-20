# coding:utf-8
import numpy as np
import sys
from net import Network
import tensorflow as tf

reload(sys)

import cv2


def transform(age, gender):
    if gender == 1:
        new_gen = 'M'
    else:
        new_gen = 'F'

    if age == 0:
        new_age = '(20,30)'  # random.randint(20,30)
    elif age == 1:
        new_age = '(31,45)'  # random.randint(31,45)
    elif age == 2:
        new_age = '(46,50)'  # random.randint(46,50)
    else:
        new_age = '(51,65)'  # random.randint(51,65)
    return new_gen, new_age


vc = cv2.VideoCapture('/home/htkang/bigdata/age_gender/data/edited.mp4')  # read video
c = 1

# initialize network
net = Network(
    n_output_gen=2,
    n_output_age=4,
    n_length=128,
    learning_rate=0.01,
    batch_size=32,
    channel=1,
    output_graph=False,
    use_ckpt=True
)

if vc.isOpened():  # determine whether it opened
    rval, frame = vc.read()
else:
    rval = False

timeF = 10  # frequency
face_cascade = cv2.CascadeClassifier('/home/htkang/opencv/data'
                                     '/haarcascades/haarcascade_frontalface_default.xml')
while rval:  # read each frame
    rval, frame = vc.read()
    if (c % timeF == 0):  #
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(5, 5),
                                              flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
        if len(faces) == 0:
            continue

        gray1 = np.copy(gray)

        for (x, y, w, h) in faces:
            # cv2.circle(image,((x+x+w)/2,(y+y+h)/2),w/2,(0,255,0),2)
            roiimage = gray1[y:y + h, x:x + w]
            roiimage = cv2.resize(roiimage, (128, 128), interpolation=cv2.INTER_CUBIC)

            test_gray_1 = tf.expand_dims(roiimage, 0)  # add dimension
            test_gray_2 = tf.expand_dims(test_gray_1, -1)
            with tf.Session() as sess:
                input_image = sess.run(test_gray_2)

            gender, age = net.get_result(input_image)
            print ("gender is:%d, age is:%d " % (gender, age))

            gender, age = transform(age, gender)
            font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, hscale= 1.0, vscale= 1.0, shear=0, thickness=2, lineType=8)  # 1,1,0,0,1
            cv2.cv.PutText(cv2.cv.fromarray(frame), "Age: " + age + " " + "Gn: " + str(gender), (30, 30), font, (0, 255, 0))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 6
            cv2.imshow("Faces found", frame)  # 7
            cv2.waitKey(1)  # 8
    c = c + 1
    cv2.waitKey(1)
vc.release()



