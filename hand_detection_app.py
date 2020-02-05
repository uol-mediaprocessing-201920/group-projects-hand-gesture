import keras
import cv2
import numpy as np
from keras.preprocessing import image
from pynput.keyboard import Key, Controller

# the 6 gestures which can be predict by the model
gesture = ["palm", "peace", "thumb", "fist", "ok", "L"]

# load the model
model = keras.models.load_model('handrecognition_model_backSub2.h5')

cam = cv2.VideoCapture(1)
keyboard = Controller()

kernel = np.ones((3, 3), np.uint8)
gestureActiveCounter = 0
lastChar = None
backgroundCaptured = False
readyForAction = True

while True:
    # read the webcam and get the roi
    frame = cam.read()[1]
    frame = cv2.flip(frame, 1)
    roi = frame[100:400, 300:600]
    roi_norm = roi.copy()
    # draw a green rectangle, to show the box where you can make the gestures
    cv2.rectangle(frame, (299, 99), (600, 400), (0, 255, 0), 0)

    if (backgroundCaptured):
        cv2.destroyWindow("Webcam")

        # create the binary picture of the hand
        roi = cv2.bilateralFilter(roi, 5, 50, 100)
        mask = backSub.apply(roi, learningRate=0)
        mask = cv2.erode(mask, kernel, iterations=1)
        res = cv2.bitwise_and(roi, roi, mask=mask)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        ret, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_norm = thresh.copy()

        # prepare the picture for the model
        data = cv2.resize(thresh, (100, 100))
        predict_image = image.img_to_array(data)
        predict_image = np.expand_dims(predict_image, axis=0)

        # create the output window
        thresh = cv2.resize(thresh, (480, 480))
        thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        concatFrame = np.hstack((frame, thresh))

        # if a hand is detected in the roi
        if sum(sum(data)) > 3000:
            prediction = model.predict(predict_image)
            print(np.argmax(prediction))
            print(100 * np.max(prediction))
            print("*********************")

            # if the gesture was recognized with over 90% safeness, output the prediction
            if np.max(prediction) > 0.9:

                # it the gesture is the fist
                if np.argmax(prediction) == 3:
                    readyForAction = True

                # if the gesture is a palm
                if np.argmax(prediction) == 0 and readyForAction:
                    readyForAction = False
                    keyboard.press(Key.right)
                    keyboard.release(Key.right)
                    # keyboard.press(Key.down)
                    # keyboard.release(Key.down)

                # if the gesture is the L
                if np.argmax(prediction) == 5 and readyForAction:
                    readyForAction = False
                    keyboard.press(Key.left)
                    keyboard.release(Key.left)
                    # keyboard.press(Key.up)
                    # keyboard.release(Key.up)

                # set text on the output window with the predicted gesture
                cv2.putText(concatFrame, gesture[np.argmax(prediction[0])], (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.8,
                            (0, 0, 255), 3, cv2.LINE_AA)
                lastChar = gesture[np.argmax(prediction[0])]
            else:
                cv2.putText(concatFrame, lastChar, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 3, cv2.LINE_AA)

        # if no hand is in the roi
        else:
            cv2.putText(concatFrame, "nothing", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 3,
                        cv2.LINE_AA)
            lastChar = "nothing"
            readyForAction = True

        cv2.imshow("Gesture Detection", concatFrame)

    else:
        cv2.putText(frame, "Press 'b' to capture background", (65, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)

    cv2.imshow("Webcam", frame)
    key = cv2.waitKey(100)

    # if pressed "Esc" the program will be closed
    if key == 27:
        cv2.destroyWindow("Gesture Detection")
        break

    # if pressed b, the background will be captured
    if key == ord('b'):
        backgroundCaptured = True
        backSub = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=100, detectShadows=True)

    # if pressed r, the background will be resetted
    if key == ord('r'):
        backgroundCaptured = False

    # if pressed s, the picture of the roi and the mask will be saved
    if key == ord('s'):
        cv2.imwrite("Norm.jpg", roi_norm)
        cv2.imwrite("Mask.jpg", thresh_norm)
