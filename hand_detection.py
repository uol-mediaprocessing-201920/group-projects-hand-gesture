import keras
import cv2
import numpy as np
from keras.preprocessing import image
from pynput.keyboard import Key, Controller
gesture = ["palm", "peace", "thumb", "fist", "ok", "L"]

model = keras.models.load_model('handrecognition_model_backSub.h5')
kernel = np.ones((3,3), np.uint8)
cam = cv2.VideoCapture(1)
lastChar = None
gestureActiveCounter = 0
backgroundCaptured = False
isFist = True

keyboard = Controller()

while True:
    frame = cam.read()[1]
    frame = cv2.flip(frame, 1)
    roi = frame[100:400, 300:600]
    cv2.rectangle(frame, (299, 99), (600, 400), (0, 255, 0), 0)

    if (backgroundCaptured):

        roi = cv2.bilateralFilter(roi, 5, 50, 100)
        mask = backSub.apply(roi, learningRate=0)
        mask = cv2.erode(mask, kernel, iterations=1)
        res = cv2.bitwise_and(roi, roi, mask=mask)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (41, 41), 0)
        ret, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        data = cv2.resize(thresh, (100, 100))
        predict_image = image.img_to_array(data)
        predict_image = np.expand_dims(predict_image, axis=0)
        if(sum(sum(data))>3000):
            prediction = model.predict(predict_image)
            print(np.argmax(prediction))
            print(100*np.max(prediction))
            print("*********************")
            cv2.imshow("ROI", thresh)
            if(np.max(prediction)>0.9):
                if(lastChar == gesture[np.argmax(prediction[0])]):
                    gestureActiveCounter+=1
                else:
                    gestureActiveCounter = 0

                if np.argmax(prediction) == 3:
                    isFist = True
                if np.argmax(prediction) == 1 and isFist:
                    isFist = False
                    keyboard.press(Key.right)
                    keyboard.release(Key.right)
                    keyboard.press(Key.down)
                    keyboard.release(Key.down)
                if np.argmax(prediction) == 5 and isFist:
                    isFist = False
                    keyboard.press(Key.left)
                    keyboard.release(Key.left)
                    keyboard.press(Key.up)
                    keyboard.release(Key.up)

                cv2.putText(frame, gesture[np.argmax(prediction[0])], (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
                lastChar = gesture[np.argmax(prediction[0])]
            else:
                cv2.putText(frame, lastChar, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            cv2.putText(frame, "nothing", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3,
                        cv2.LINE_AA)
            lastChar = "nothing"
    cv2.imshow("Webcam", frame)


    key = cv2.waitKey(100)
    if key == 27:
        cv2.destroyWindow("Gesture Detection")
        break
    if key == ord('b'):
        backgroundCaptured = True
        backSub = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=200, detectShadows=True)
    if key == ord('r'):
        backgroundCaptured = False
