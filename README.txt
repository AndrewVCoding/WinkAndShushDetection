For both detectors, I adjusted the parameters used for the cascades, tweaking them until I was getting fewer false
positives/negatives.

Wink Detection
    I added cascades for both left and right eyes, and use the difference between the number of each detected on the
    face to determine if the face is winking. I also factor in the result of the wink cascade, so that if either value
    indicates the face is winking, then the program counts it as a wink, but with more weight given to the left/right
    eye detection.
    When running detection on video, I included a confidence value that increases when a wink is detected. The program
    uses the confidence when determining whether to draw a blue rectangle to indicate a detected wink. The confidence
    declines over time when the program is not detecting a wink. This helps avoid missing winks in subsequent frames,
    where they are more likely to occur, and smooths out the detection.
    I also changed the ROI for the eye cascades to only look at the top portion of a face.

Sush Detection
    I added a finger detection cascade that uses detected mouth regions as its ROI. This way, the program won't falsely
    detect a shush gesture just from the presence of fingers in front of a face.
    fingerCascade.xml came from here:
        https://github.com/PrafulBhosale/Finger-Detection-Application/blob/master/DetectFingerProject/cascade.xml
