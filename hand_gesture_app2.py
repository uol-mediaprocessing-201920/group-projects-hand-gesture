import cv2
import numpy as np
from sklearn.metrics import pairwise
import keras
import time
from keras.preprocessing import image

# Paramters
PROGRAM_TITLE = 'Hand Gesture Recognition'
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 500
CAMERA_PORT = 0  # 0 bei Mac OS X, 1 bei Windows
REGION_X = 0.5
REGION_Y = 0.7
LEARNING_RATE = 0
blur_value = 5
parse_threshold = 2
gesture = ["palm", "peace", "thumb", "fist", "ok", "L"]
model = keras.models.load_model('handrecognition_model_backSub.h5')

# FLAGS
FIRST_ITERATION = True

# Text Paramters
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 2
fontColor = (0, 0, 255)
lineType = 3

# Camera
camera = cv2.VideoCapture(CAMERA_PORT)

# Window
cv2.namedWindow(PROGRAM_TITLE, cv2.WINDOW_NORMAL)
cv2.resizeWindow(PROGRAM_TITLE, 600, 600)

# Globals
bg_model = None
parsed_img = None


def quit_program():
    """
    Diese Funktion schließt das Programm.
    """
    camera.release()
    cv2.destroyAllWindows()


def capture_background():
    """
    Diese Funktion speichert den derzeitigen Hintergrund ab.
    """
    global bg_model
    bg_model = cv2.createBackgroundSubtractorMOG2(0, 50)

def create_snapshot():
    """
    Diese Methode erstellt ein Bild vom derzeitigen analysieten Bild.
    """
    global parsed_img
    if parsed_img is not None:
        cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(parsed_img, cv2.COLOR_RGB2BGR))


def parse_key_command():
    """
    Diese Funktion verlinkt Keys mit ihren dazugehörigen Actions.
    """
    key = cv2.waitKey(1)
    action = {
        ord('q'): quit_program,
        ord('b'): capture_background,
        ord('s'): create_snapshot
    }
    func = action.get(key, None)
    if func is not None:
        print('Executing action: \''+func.__name__+'\'')
        func()


def crop_img(frame):
    height, width, _ = frame.shape
    y = 0
    x = int(REGION_X * width)
    h = int(REGION_Y * height)
    w = width - x
    roi = frame[y:y + h, x:x + w]
    return roi


def remove_background(frame):
    global bg_model, LEARNING_RATE
    # Berechne Maske
    mask = bg_model.apply(frame, learningRate=LEARNING_RATE)
    # Erosion, um rauschen zu entfernern
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    # Schneide Maske aus Bild
    res = cv2.bitwise_and(frame, frame, mask=mask)
    return res


def parse_img(img):
    global blur_value, parse_threshold
    # Bilateral Filter
    filter = cv2.bilateralFilter(img, 5, 50, 100)
    # Grayscale
    gray = cv2.cvtColor(filter, cv2.COLOR_BGR2GRAY)
    # Make the grey scale image have three channels
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # Blur
    blur = cv2.GaussianBlur(gray, (blur_value, blur_value), 0)
    # Threshold
    thresh = cv2.threshold(blur, parse_threshold, 255, cv2.THRESH_BINARY)[1]
    # Erosion
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(thresh, kernel, iterations=1)
    return erosion


def draw_convex_hull(img):
    thresh1 = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    maxArea = -1
    res = None
    if length > 0:
        for i in range(length):
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > maxArea:
                maxArea = area
                ci = i

        res = contours[ci]
        hull = cv2.convexHull(res)
        img = np.zeros(img.shape, np.uint8)
        cv2.drawContours(img, [res], 0, (0, 255, 0), 2)
        cv2.drawContours(img, [hull], 0, (0, 0, 255), 3)
    return img, res


def count_fingers(img):
    thresh = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return 0

    segmented = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(segmented)

    extreme_top = tuple(hull[hull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(hull[hull[:, :, 1].argmax()][0])
    extreme_left = tuple(hull[hull[:, :, 0].argmin()][0])
    extreme_right = tuple(hull[hull[:, :, 0].argmax()][0])

    # Zentrum
    cX = int((extreme_left[0] + extreme_right[0]) / 2)
    cY = int((extreme_top[1] + extreme_bottom[1]) / 2)

    distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    maximum_distance = distance[distance.argmax()]

    radius = int(0.8 * maximum_distance)
    circumference = (2 * np.pi * radius)
    circular_roi = np.zeros(thresh.shape[:2], dtype="uint8")
    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)
    circular_roi = cv2.bitwise_and(thresh, thresh, mask=circular_roi)
    cnts, _ = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    count = 0
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
            count += 1

    return count


# Main loop
while camera.isOpened():
    ret, frame = camera.read()
    height, width, _ = frame.shape

    # Zeichne ein Rechteck für den Aktionsbereich
    cv2.rectangle(frame, (int(REGION_X * width), 0), (width, int(REGION_Y * height)), (255, 0, 0), 2)

    # Füge Text zum Original-Bild hinzu
    original_image = frame.copy()
    cv2.putText(original_image, "Original", (10, 50), font, fontScale, fontColor, lineType, cv2.LINE_AA)

    # Berechne ROI
    roi = crop_img(frame)
    roi_height, roi_width, _ = roi.shape

    # Prediction
    prediction_img = roi.copy()

    # Resize das Original-Bild, um es neben den anderen Bildern darstellen zu können
    original_image = cv2.resize(original_image, (roi_width, roi_height))

    if bg_model is not None:
        roi = remove_background(roi)
        parsed_img = parse_img(roi)

        # 'Manuelle' Methode: Konvex-Hülle
        finger_count = count_fingers(parsed_img.copy())
        roi, _ = draw_convex_hull(parsed_img.copy())
        if finger_count is not None:
            cv2.putText(roi, "Finger: "+str(finger_count), (10, 50), font, fontScale, fontColor, lineType, cv2.LINE_AA)

        # Methode mit Machine Learning
        prediction_img = cv2.cvtColor(prediction_img, cv2.COLOR_BGR2GRAY)
        prediction_img = cv2.flip(prediction_img, 1)
        data = cv2.resize(prediction_img, (100, 100))
        prediction_img = image.img_to_array(data)
        prediction_img = np.expand_dims(prediction_img, axis=0)
        prediction = model.predict(prediction_img)
        gesture_string = gesture[np.argmax(prediction[0])]
        prediction_img = parsed_img.copy()
        cv2.putText(prediction_img, gesture_string, (10, 50), font, fontScale, fontColor, lineType, cv2.LINE_AA)

    # Zeige Alle Bilder
    cv2.imshow(PROGRAM_TITLE, np.hstack((original_image, roi, prediction_img)))

    # Überprüfe, ob eine Taste gedrückt wurde
    parse_key_command()

    if FIRST_ITERATION:
        FIRST_ITERATION = False
    else:
        # Vergrößere das Fenster
        cv2.resizeWindow(PROGRAM_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT)
