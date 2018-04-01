import numpy as np
import cv2
import math
import os
from os import listdir
from os.path import isfile, join
import sys


# Used to detect features using the given cascade
# Returns a tuple containing the rectangles around the features
def detectFeature(frame, location, ROI, cascade):
    features = cascade.detectMultiScale(ROI, 1.15, 3, 0, (20, 20))
    rectangles = []
    for (mx, my, mw, mh) in features:
        mx += location[0]
        my += location[1]
        cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)
        rectangles.append((mx, my, mw, mh))
    return rectangles


def detect(frame, faceCascade, mouthsCascade, fingerCascade):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # gray_frame = cv2.equalizeHist(gray_frame)
    # gray_frame = cv2.medianBlur(gray_frame, 5)

    faces = faceCascade.detectMultiScale(
        gray_frame, 1.15, 4, 0 | cv2.CASCADE_SCALE_IMAGE, (20, 20))
    detected = 0
    for (x, y, w, h) in faces:
        # ROI for feature
        x1 = x
        h2 = int(h / 2)
        y1 = y + h2
        mouthROI = gray_frame[y1:y1 + h2, x1:x1 + w]

        mouths = detectFeature(frame, (x1, y1), mouthROI, mouthsCascade)

        # Look for fingers in the mouth area
        for (xx, yy, ww, hh) in mouths:
            # ROI for feature
            x2 = xx
            h3 = int(hh / 2)
            y2 = yy + h2
            fingerROI = gray_frame[y2:y2 + h3, x2:x2 + ww]
            fingerFrame = frame
            fingers = detectFeature(fingerFrame, (x2, y2), fingerROI, fingerCascade)

        if len(mouths) == 0 or (len(mouths) >= 1 and len(fingers) >= 1):
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
    totalCnt = 0
    for f in files:
        img = cv2.imread(f)
        if type(img) is np.ndarray:
            lCnt = detect(img, cascade1, cascade2, cascade3)
            totalCnt += lCnt
            if windowName != None:
                cv2.destroyWindow(windowName)
            windowName = f
            cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(windowName, img)
            cv2.waitKey(0)
    return totalCnt


def runonVideo(face_cascade, eyes_cascade, fingerCascade):
    videocapture = cv2.VideoCapture(0)
    if not videocapture.isOpened():
        print("Can't open default video camera!")
        exit()

    windowName = "Live Video"
    showframe = True
    while (showframe):
        ret, frame = videocapture.read()

        if not ret:
            print("Can't capture frame")
            break
        detect(frame, face_cascade, eyes_cascade, fingerCascade)
        cv2.imshow(windowName, frame)
        if cv2.waitKey(30) >= 0:
            showframe = False

    videocapture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # check command line arguments: nothing or a folderpath
    if len(sys.argv) != 1 and len(sys.argv) != 2:
        print(sys.argv[0] + ": got " + len(sys.argv) - 1 +
              "arguments. Expecting 0 or 1:[image-folder]")
        exit()

    # load pretrained cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    mouth_cascade = cv2.CascadeClassifier('Mouth.xml')
    finger_cascade = cv2.CascadeClassifier('cascades/fingerCascade.xml')

    if (len(sys.argv) == 2):  # one argument
        folderName = sys.argv[1]
        detections = run_on_folder(face_cascade, mouth_cascade, finger_cascade, folderName)
        print("Total of ", detections, "detections")
    else:  # no arguments
        runonVideo(face_cascade, mouth_cascade, finger_cascade)
