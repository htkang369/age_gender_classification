import cv2
import numpy as np
import sys
# Get user supplied values
imagePath = "/home/htkang/bigdata/age_gender/data/she.jpg"
# Create the haar cascade
faceCascade = cv2.CascadeClassifier("/home/htkang/opencv/data/haarcascades/"
                                    "haarcascade_frontalface_alt.xml") #1
# Read the image
image = cv2.imread(imagePath)#2
print image.shape


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#3
#cv2.equalizeHist(gray, gray)
# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.15,
    minNeighbors=5,
    minSize=(30,30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
) #4
print "Found {0} faces!".format(len(faces))#5
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2) #6
cv2.imshow("Faces found", image)#7
cv2.waitKey(5000) #8