import cv2
import argparse
import sys
import imutils
import numpy as np
import os
from neural_net.cnn import LeNet
import tensorflow as tf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True, help="Path to input video")
    ap.add_argument("-m", "--model", required=True, help="Path to trained model")
    args = vars(ap.parse_args())

    path_to_video = args["video"]
    if not os.path.exists(path_to_video):
        raise ValueError("Path {} to input video does not exist".format(path_to_video))

    path_to_model = args["model"]
    if not os.path.exists(path_to_model):
        raise ValueError("Path {} to trained model does not exist".format(path_to_video))

    cap = cv2.VideoCapture(path_to_video)
    if not cap.isOpened():
        print("[ERROR]: Could not open the video specified by {} path".format(path_to_video))
        sys.exit()

    model = LeNet.build(width=32,height=32,depth=1, classes=2)
    model = tf.keras.models.load_model(path_to_model)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR]: Can't receive frame (stream end?). Exiting ...")
            sys.exit()

        frame = np.array(frame)
        height = frame.shape[0]
        width = frame.shape[1]

        crop_img = frame[int(height/5)-14:height - (int(height/9)), int(width/6):width-int(width/6)]

        low = np.array([20, 100, 100])
        high = np.array([42, 255, 255])

        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

        img_mask = cv2.inRange(hsv, low, high)
        img_mask_blured = cv2.medianBlur(img_mask, 5)

        roi_arr = []
        cnts = cv2.findContours(img_mask_blured.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        for (i, c) in enumerate(cnts, 1):
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print("Center of the rectangle is ({},{})".format(cX, cY))

            x,y,w,h = cv2.boundingRect(c)
            print("Top-left coordinate of the rectangle is ({},{})".format(x,y))
            print("Width and height of the rectangle are {} and {}".format(w,h))

            roi = tuple((crop_img[y:y+h, x:x+w], (cX,cY)))

            gray = cv2.cvtColor(roi[0], cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (32, 32))
            gray = np.expand_dims(np.array(gray), 2)

            #cv2.imshow("RoI {}".format(i),roi[0])
            print(model.predict(np.expand_dims(gray,0), batch_size=1, verbose=0));

            roi_arr.append(roi)

    cap.release()
    cv2.destroyAllWindows()
