import cv2
import numpy as np
import math

# Paramters
PROGRAM_TITLE = 'Hand Gesture Recognition'
REGION_X = 0.5
REGION_Y = 0.7
LEARNING_RATE = 0
blur_value = 3
parse_threshold = 25

# Text Paramters
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0, 0, 0)
lineType = 2


# Camera
camera = cv2.VideoCapture(0)

# Window
cv2.namedWindow(PROGRAM_TITLE, cv2.WINDOW_NORMAL)
cv2.resizeWindow(PROGRAM_TITLE, 600, 600)

# Globals
bg_model = None


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


def parse_key_command():
    """
    Diese Funktion verlinkt Keys mit ihren dazugehörigen Actions.
    """
    key = cv2.waitKey(1)
    action = {
        ord('q'): quit_program,
        ord('b'): capture_background
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
    # Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Make the grey scale image have three channels
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Blur
    img = cv2.GaussianBlur(img, (blur_value, blur_value), 0)
    img = cv2.threshold(img, parse_threshold, 255, cv2.THRESH_BINARY)[1]
    return img


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


def calculateFingers(res, drawing):  # -> finished bool, cnt: finger count
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if defects is not None:
            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            return True, cnt
    return False, 0


# Main loop
while camera.isOpened():
    ret, frame = camera.read()
    height, width, _ = frame.shape

    # Zeichne ein Rechteck für den Aktionsbereich
    cv2.rectangle(frame, (int(REGION_X * width), 0), (width, int(REGION_Y * height)), (255, 0, 0), 2)

    # Füge Text zum Original-Bild hinzu
    original_image = frame.copy()
    cv2.putText(original_image, "Original", (10, 30), font, fontScale, fontColor, lineType)

    # Berechne ROI
    roi = crop_img(frame)
    roi_height, roi_width, _ = roi.shape

    # Resize das Original-Bild, um es neben den anderen Bildern darstellen zu können
    original_image = cv2.resize(original_image, (roi_width, roi_height))

    if bg_model is not None:
        roi = remove_background(roi)
        parsed_img = parse_img(roi)
        roi = parsed_img.copy()

        # 'Manuelle' Methode: Konvex-Hülle
        roi, res = draw_convex_hull(roi)
        #count = calculateFingers(res, roi)
        #print(count)




    # Zeige Alle Bilder
    cv2.imshow(PROGRAM_TITLE, np.hstack((original_image, roi)))

    # Überprüfe, ob eine Taste gedrückt wurde
    parse_key_command()
