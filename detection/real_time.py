# coding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import skimage

path ="/home/htkang/bigdata/age_gender/data/FroggerHighway.avi"
# reader = imageio.get_reader('<video0>')
# for i, im in enumerate(reader):
#     print('Mean of frame %i is %1.1f' % (i, im.mean()))
#     image = skimage.img_as_float(im).astype(np.float64)
#     print image.shape
#     # opImg = skimage.img_as_ubyte(im, True)
#     # print opImg.shape
#     plt.imshow(image)
#     plt.axis('off')
#     plt.show()
#
#     if i==5:break




cap = cv2.VideoCapture(path)

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''
print cap.isOpened()
# cap = cv2.VideoCapture("test.mp4") #open video
success, frame = cap.read()
# classifier = cv2.CascadeClassifier("/Users/yuki/anaconda/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml")

# haarcascade_frontalface_default.xml
classifier = cv2.CascadeClassifier("/home/htkang/opencv/data/haarcascades/"
                                    "haarcascade_frontalface_alt.xml") #1

while success:
    success, frame = cap.read()
    size = frame.shape[:2]
    image = np.zeros(size, dtype=np.float16)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(image, image)
    divisor = 8
    h, w = size
    minSize = (w // divisor, h // divisor)
    faceRects = classifier.detectMultiScale(image, 1.2, 2, cv2.CASCADE_SCALE_IMAGE, minSize)
    if len(faceRects) > 0:
        for faceRect in faceRects:
            x, y, w, h = faceRect
            cv2.rectangle(frame, (x, y), (x + h, y + w), (0, 255, 0), 2)
            #
            #             cv2.circle(frame, (x + w // 4, y + h // 4 + 30), min(w // 8, h // 8), (255, 0, 0))   # left eye
            #             cv2.circle(frame, (x + 3 * w //4, y + h // 4 + 30), min(w // 8, h // 8), (255, 0, 0))   #right eye
            #             cv2.rectangle(frame, (x + 3 * w // 8, y + 3 * h // 4), (x + 5 * w // 8, y + 7 * h // 8), (255, 0, 0))   #mouth
    cv2.imshow("test", frame)
    key = cv2.waitKey(10)
    c = chr(key & 255)
    if c in ['q', 'Q', chr(27)]:
        break
cv2.destroyWindow("test")'''