# import the opencv library
import cv2
import numpy as np
from tensorflow import keras
from keras.models import load_model
from imutils.perspective import four_point_transform
from imutils import contours
import matplotlib
import imutils

# image
# img = cv2.imread("photo.jpg")
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# define a video capture object
vid = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
WIDTH = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
HEIGHT = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Recognition models
# Nmber recognition
model = load_model("digits.h5")


def prediction(image, model):
    img = cv2.resize(image, (28, 28))
    img = img / 255
    img = img.reshape(1, 28, 28, 1)
    predict = model.predict(img, verbose=None)
    prob = np.amax(predict)
    class_index = np.argmax(model.predict(img, verbose=None), axis=1)
    result = class_index[0]
    if prob < 0.75:
        result = 0
        prob = 0
    return result, prob


while True:
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    frame_copy = frame.copy()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    frame_copy = cv2.flip(frame_copy, 1)
    frame_gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(frame_gray, (9, 9), 0)
    edged = cv2.Canny(blurred, 50, 70, 255)

    """
    # find contours in the edge map, then sort them by their
    # size in descending order
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    peri = cv2.arcLength(cnts[0], True)
    displayCnt = cv2.approxPolyDP(cnts[0], 0.02 * peri, True)

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if the contour has four vertices, then we have found
        # the thermostat display
        if len(approx) == 4:
            displayCnt = approx
            break"""

    # extract the thermostat display, apply a perspective transform
    # to it
    # warped = four_point_transform(frame_gray, displayCnt.reshape(4, 2))
    # cv2.imshow("warped", warped)

    thresh = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[
        1
    ]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Display the resulting frame
    cv2.imshow("frame", edged)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
