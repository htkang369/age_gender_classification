# coding:utf-8
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf8')


import cv2

# 待检测的图片路径
imagepath = '/home/lee/bigdata/photo.jpeg'

# 获取训练好的人脸的参数数据，这里直接从GitHub上使用默认值

face_cascade = cv2.CascadeClassifier('/home/lee/opencv-2.4.9/data/haarcascades/haarcascade_frontalface_default.xml')

# 读取图片

image = cv2.imread(imagepath)
#print image
#a=np.shape(image)
#print a
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
print gray
print type(gray)
i=1
# 探测图片中的人脸
faces = face_cascade.detectMultiScale(gray,scaleFactor=1.15,minNeighbors=5,minSize=(5,5),flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

print ("found {0} faces!".format(len(faces)))

image1=np.copy(image)

for (x,y,w,h) in faces:
    cv2.rectangle(image1,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imwrite('./'+'photo1.jpeg',image1)

cv2.imshow("Find Faces!",image1)
cv2.waitKey(10000)

for (x,y,w,h) in faces:
   #cv2.circle(image,((x+x+w)/2,(y+y+h)/2),w/2,(0,255,0),2)   
    roiimage=image[y:y+h,x:x+w]
    roiimage=cv2.resize(roiimage,(128,128),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('./'+str(i)+'.jpeg',roiimage)
    #name=cv2.NamedWindow('extracted face'+str(i))
    cv2.imshow('extracted face'+str(i),roiimage)
    cv2.waitKey(i*10000)
    i+=1




