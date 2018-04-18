import tensorflow as tf
from net import Network
import cv2
import numpy as np

def transform(age,gender):
    if gender == 1:
        new_gen = 'Male'
    else:
        new_gen = 'Female'

    if age==0:
        new_age = np.random.randint(18,30)
    elif age==1:
        new_age = np.random.randint(31,45)
    elif age==2:
        new_age = np.random.randint(46,50)
    else:
        new_age = np.random.randint(51,65)
    return new_gen,new_age


if __name__ =="__main__":

    # import test images
    index = 5
    imagepath = '/home/htkang/bigdata/age_gender/data/'+str(index)+'.jpg'
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier("/home/htkang/opencv/data/haarcascades/"
                                        "haarcascade_frontalface_alt.xml")  # 1
    image_in = cv2.cv.LoadImage(imagepath)  # 2
    image = np.asarray(image_in[:,:])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 3
    # cv2.equalizeHist(gray, gray)
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )  # 4
    # print "Found {0} faces!".format(len(faces))  # 5
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 6
    # cv2.imshow("Faces found", image)  # 7
    # cv2.waitKey(5000)  # 8

    i=1
    for (x, y, w, h) in faces:
        # cv2.circle(image,((x+x+w)/2,(y+y+h)/2),w/2,(0,255,0),2)
        roiimage = gray[y:y + h, x:x + w]
        roiimage = cv2.resize(roiimage, (128, 128), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('./' + str(index) + '.jpg', roiimage)
        #cv2.imshow('extracted face' + str(i), roiimage)
        #cv2.waitKey(i * 200)
        i += 1

    new_path = '/home/htkang/bigdata/age_gender/code/joint/'+str(index)+'.jpg'
    test_image = cv2.imread(new_path)
    test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY) # test image

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
    test_gray_1 = tf.expand_dims(test_gray, 0) # add dimension
    test_gray_2 = tf.expand_dims(test_gray_1, -1)
    with tf.Session() as sess:
        input_image = sess.run(test_gray_2)


    gender,age = net.get_result(input_image)

    print ("gender is:%d, age is:%d " %(gender,age))

    gender,age = transform(age,gender)

    font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 1, 0, 0, 1)
    cv2.cv.PutText(image_in, "Age: "+str(age)+" " +"Gender: "+str(gender), (30, 30), font, (0, 255, 0))

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 6
    cv2.imshow("Faces found", image)  # 7
    cv2.waitKey(5000)  # 8