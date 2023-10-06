# import the opencv library
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
from picamera.array import PiRGBArray
from picamera import PiCamera
import RPi.GPIO as GPIO
#from tensorflow import keras
#from keras.models import load_model

GPIO.setmode(GPIO.BOARD)
#GPIO.setup(12, GPIO.OUT)



# define a video capture object
camera = PiCamera()
camera.resolution = (640, 480)
WIDTH = 640
HEIGHT = 480
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# Recognition models
# Nmber recognition
model = Interpreter("TFlite_digits.tflite")
model.allocate_tensors()



def prediction(image, model):

    input_details = model.get_input_details()
    output_details = model.get_output_details()
    
    input_shape = input_details[0]['shape'][1:3] # (28,28)
    img = cv2.resize(image, input_shape)
    img = img / 255
    img = img.reshape(1, 28, 28, 1)

    model.set_tensor(input_details[0]['index'], np.float32(img))
    model.invoke()
    predict = model.get_tensor(output_details[0]['index'])
    
    prob = np.amax(predict)
    class_index = np.argmax(predict)
    result = class_index
    if prob < 0.75:
        result = 0
        prob = 0
    return result, prob


for frame in camera.capture_continuous(rawCapture, format="rgb", use_video_port=True):
    # Capture the video frame
    # by frame
    frame_copy = frame.array
    frame_copy = cv2.rotate(frame_copy, cv2.ROTATE_180)
    #frame_copy = cv2.flip(frame_copy, 1)
    frame_gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)

    # Set box size
    bbox_size = (60*2, 60*2)
    # Set box top left corner X and Y positions
    bbox = [
        (int(WIDTH // 2 - bbox_size[0] // 2), int(HEIGHT // 2 - bbox_size[1] // 2)),
        (int(WIDTH // 2 + bbox_size[0] // 2), int(HEIGHT // 2 + bbox_size[1] // 2)),
    ]

    img_cropped = frame_copy[bbox[0][1] : bbox[1][1], bbox[0][0] : bbox[1][0]]
    img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (200, 200))
    cv2.imshow("guessbox", img_gray)

    result, probability = prediction(img_gray, model)
    color= (255,255,255)
    
    cv2.putText(
        frame_copy,
        f"Siffra: {result}",
        (40, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame_copy,
        "Sannolikhet: " + "{:.2f}".format(probability),
        (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )
    
    # Set GPIO pins
    #GPIO.output(12, 1)
    #GPIO.output(12, 0)

    #GPIO data ut
    #GPIO.setmode(GPIO.BCM)
    GPIO.setup(3, GPIO.OUT) #A1
    GPIO.setup(5, GPIO.OUT) #A2
    GPIO.setup(7, GPIO.OUT) #A3
    GPIO.setup(11, GPIO.OUT) #A0
    
    if (result == 1):
        GPIO.output(7, 0)  #A3
        GPIO.output(5, 0)  #A2       
        GPIO.output(3, 0)  #A1
        GPIO.output(11, 1) #A0
        
    elif (result == 2):
        GPIO.output(7, 0)  #A3
        GPIO.output(5, 0)  #A2       
        GPIO.output(3, 1)  #A1
        GPIO.output(11, 0) #A0
        
    elif (result == 3):
        GPIO.output(7, 0)  #A3
        GPIO.output(5, 0)  #A2       
        GPIO.output(3, 1)  #A1
        GPIO.output(11, 1) #A0
        
    elif (result == 4):
        GPIO.output(7, 0)  #A3
        GPIO.output(5, 1)  #A2       
        GPIO.output(3, 0)  #A1
        GPIO.output(11, 0) #A0
        
    elif (result == 5):
        GPIO.output(7, 0)  #A3
        GPIO.output(5, 1)  #A2       
        GPIO.output(3, 0)  #A1
        GPIO.output(11, 1) #A0
        
    elif (result == 6):
        GPIO.output(7, 0)  #A3
        GPIO.output(5, 1)  #A2       
        GPIO.output(3, 1)  #A1
        GPIO.output(11, 0) #A0
        
    elif (result == 7):
        GPIO.output(7, 0)  #A3
        GPIO.output(5, 1)  #A2       
        GPIO.output(3, 1)  #A1
        GPIO.output(11, 1) #A0
        
    elif (result == 8):
        GPIO.output(7, 1)  #A3
        GPIO.output(5, 0)  #A2       
        GPIO.output(3, 0)  #A1
        GPIO.output(11, 0) #A0
        
    elif (result == 9):
        GPIO.output(7, 1)  #A3
        GPIO.output(5, 0)  #A2       
        GPIO.output(3, 0)  #A1
        GPIO.output(11, 1) #A0
    else :
        GPIO.output(7, 0)  #A3
        GPIO.output(5, 0)  #A2       
        GPIO.output(3, 0)  #A1
        GPIO.output(11, 0) #A0

    # Draw a rectangle
    cv2.rectangle(frame_copy, bbox[0], bbox[1], (0, 255, 0), 3)

    # Display the resulting frame
    cv2.imshow("frame", frame_copy)
    
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# After the loop release the cap object
# Destroy all the windows
cv2.destroyAllWindows()
GPIO.cleanup()
