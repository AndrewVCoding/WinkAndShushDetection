import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
import sys


# Detect Feature
def detectSingularFeature(frame, location, ROI, cascade, size):
    features = cascade.detectMultiScale(
        ROI, 1.15, 3, 0 | cv2.CASCADE_SCALE_IMAGE, size)
    for f in features:
        f[0] += location[0]
        f[1] += location[1]
        x, y, w, h = f[0], f[1], f[2], f[3]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 60), 2)

    # Return a positive value if only one feature is detected, otherwise a negative value
    if len(features) == 1:
        return 1
    else:
        return -1


def detectWink(frame, location, ROI, cascade):
    eyes = cascade.detectMultiScale(
        ROI, 1.15, 3, 0 | cv2.CASCADE_SCALE_IMAGE, (20, 20))
    for e in eyes:
        e[0] += location[0]
        e[1] += location[1]
        x, y, w, h = e[0], e[1], e[2], e[3]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    if len(eyes) == 1:
        return 1
    else:
        return -1


def folderDetect(frame, faceCascade, eyesCascade):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # possible frame pre-processing:
    # gray_frame = cv2.equalizeHist(gray_frame)
    # gray_frame = cv2.medianBlur(gray_frame, 5)

    scaleFactor = 1.15  # range is from 1 to ..
    minNeighbors = 1  # range is from 0 to ..
    flag = 0 | cv2.CASCADE_SCALE_IMAGE  # either 0 or 0|cv2.CASCADE_SCALE_IMAGE
    minSize = (30, 30)  # range is from (0,0) to ..
    faces = faceCascade.detectMultiScale(
        gray_frame,
        scaleFactor,
        minNeighbors,
        flag,
        minSize)

    detected = 0
    for f in faces:
        x, y, w, h = f[0], f[1], f[2], f[3]
        faceROI = gray_frame[y:y + h, x:x + w]

        # Get features for detection
        # I want to detect the presence of the left and rights eyes seperately
        leyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades
                                            + 'haarcascade_lefteye_2splits.xml')
        reyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades
                                            + 'haarcascade_righteye_2splits.xml')
        leftEye = detectSingularFeature(frame, (x, y), faceROI, leyeCascade, (20, 20))
        rightEye = detectSingularFeature(frame, (x, y), faceROI, reyeCascade, (20, 20))
        wink = detectWink(frame, (x, y), faceROI, eyesCascade)

        # Assign higher value when "wink" is detected and when leftEye != rightEye.
        if (wink + abs(leftEye - rightEye)) >= 1:
            detected += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return detected


# A faster detection algorithm for video
def videoDetect(frame, faceCascade, eyesCascade, confidence):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # possible frame pre-processing:
    # gray_frame = cv2.equalizeHist(gray_frame)
    # gray_frame = cv2.medianBlur(gray_frame, 5)

    scaleFactor = 1.15  # range is from 1 to ..
    minNeighbors = 1  # range is from 0 to ..
    flag = 0 | cv2.CASCADE_SCALE_IMAGE  # either 0 or 0|cv2.CASCADE_SCALE_IMAGE
    minSize = (30, 30)  # range is from (0,0) to ..
    faces = faceCascade.detectMultiScale(
        gray_frame,
        scaleFactor,
        minNeighbors,
        flag,
        minSize)

    detected = 0
    for f in faces:
        x, y, w, h = f[0], f[1], f[2], f[3]
        faceROI = gray_frame[y:y + h, x:x + w]

        # Get features for detection
        # I want to detect the presence of the left and rights eyes seperately
        leyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades
                                            + 'haarcascade_lefteye_2splits.xml')
        reyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades
                                            + 'haarcascade_righteye_2splits.xml')
        leftEye = detectSingularFeature(frame, (x, y), faceROI, leyeCascade, (20, 20))
        rightEye = detectSingularFeature(frame, (x, y), faceROI, reyeCascade, (20, 20))

        # Assign higher value when "wink" is detected and when leftEye != rightEye.
        # confidence will help smooth out detection across frames by increasing the value above the threshold when
        # previous frames contained a wink.
        if (2 * abs(leftEye - rightEye) + confidence) >= 4:
            detected += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return detected


def run_on_folder(cascade1, cascade2, cascade3, folder):
    if (folder[-1] != "/"):
        folder = folder + "/"
    files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]

    windowName = None
    totalCount = 0
    for f in files:
        img = cv2.imread(f, 1)
        if type(img) is np.ndarray:
            # First try one cascade for face detection
            lCnt = folderDetect(img, cascade2, cascade3)
            # If the first one didn't find a face, try the second cascade
            if (lCnt == 0):
                lCnt += folderDetect(img, cascade1, cascade3)
            print(lCnt)
            totalCount += lCnt
            if windowName != None:
                cv2.destroyWindow(windowName)
            windowName = f
            cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(windowName, img)
            cv2.waitKey(0)
    return totalCount


def runonVideo(cascade1, cascade2, cascade3):
    videocapture = cv2.VideoCapture(0)
    if not videocapture.isOpened():
        print("Can't open default video camera!")
        exit()

    windowName = "Live Video"
    showlive = True
    # To make the detection less jumpy, smooth it out over time by keeping track of the content of the last few frames
    confidence = 0
    while (showlive):
        ret, frame = videocapture.read()

        if not ret:
            print("Can't capture frame")
            exit()

        # Use cascade2 to detect winking faces
        lCnt = videoDetect(frame, cascade2, cascade3, confidence)
        # If cascade2 didn't find a face, try cascade1. I put them in this order because I was finding more correct
        # faces with cascade2 than with cascade 1, so putting them in this order reduces processing
        # if lCnt == 0:
        #     lCnt += videoDetect(frame, cascade1, cascade3, confidence)
        confidence += 2 * lCnt
        cv2.imshow(windowName, frame)
        # Decrease the confidence level so that it doesn't get stuck at a high value
        print(confidence, ':', lCnt)
        if confidence > 5:
            confidence = confidence - 4
        else:
            confidence = confidence - 1
        if confidence < 0:
            confidence = 0
        if cv2.waitKey(30) >= 0:
            showlive = False

    # outside the while loop
    videocapture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # check command line arguments: nothing or a folderpath
    if len(sys.argv) != 1 and len(sys.argv) != 2:
        print(sys.argv[0], ": got ", len(sys.argv) - 1, "arguments. Expecting 0 or 1:[image-folder]")
        exit()

    # load pretrained cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades
                                         + 'haarcascade_frontalface_default.xml')
    face2_cascade = cv2.CascadeClassifier(cv2.data.haarcascades
                                          + 'haarcascade_frontalface_alt_tree.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades
                                        + 'haarcascade_eye.xml')

    if (len(sys.argv) == 2):  # one argument
        folderName = sys.argv[1]
        detections = run_on_folder(face_cascade, face2_cascade, eye_cascade, folderName)
        print("Total of ", detections, "detections")
    else:  # no arguments
        runonVideo(face_cascade, face2_cascade, eye_cascade)
