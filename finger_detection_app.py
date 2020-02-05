import cv2
import numpy as np
from sklearn.metrics import pairwise
import time

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
    This functions closes the program.
    """
    camera.release()
    cv2.destroyAllWindows()


def capture_background():
    """
    This function saves the current background.
    """
    global bg_model
    bg_model = cv2.createBackgroundSubtractorMOG2(0, 50)

def create_snapshot():
    """
    This method creates a snapshot of the current frame.
    """
    global parsed_img
    if parsed_img is not None:
        cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(parsed_img, cv2.COLOR_RGB2BGR))


def parse_key_command():
    """
    This function connects expected keys with their actions/functions.
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
    """
    This functions just crops the image.
    """
    height, width, _ = frame.shape
    y = 0
    x = int(REGION_X * width)
    h = int(REGION_Y * height)
    w = width - x
    roi = frame[y:y + h, x:x + w]
    return roi


def remove_background(frame):
    """
    This functions removes the background
    """
    global bg_model, LEARNING_RATE
    # calculate the mask
    mask = bg_model.apply(frame, learningRate=LEARNING_RATE)
    # use erosion to remove noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    # cut the mask out of the current frame
    res = cv2.bitwise_and(frame, frame, mask=mask)
    return res


def parse_img(img):
    """
    This function parses the image using several filters.
    """
    global blur_value, parse_threshold
    # bilateral filter to remove impurities without blurring the contours
    filter = cv2.bilateralFilter(img, 5, 50, 100)
    # grayscale
    gray = cv2.cvtColor(filter, cv2.COLOR_BGR2GRAY)
    # make the grey scale image have three channels
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # blur
    blur = cv2.GaussianBlur(gray, (blur_value, blur_value), 0)
    # threshold
    thresh = cv2.threshold(blur, parse_threshold, 255, cv2.THRESH_BINARY)[1]
    # erosion to remove last artifacts
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(thresh, kernel, iterations=1)
    return erosion


def draw_convex_hull(img):
    """
    This functions draws a convex hull around the hand segment.
    """
    thresh1 = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    max_area = -1
    res = None
    if length > 0:
        for i in range(length):
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > max_area:
                max_area = area
                ci = i

        res = contours[ci]
        hull = cv2.convexHull(res)
        img = np.zeros(img.shape, np.uint8)
        # draw the hull and the contours
        cv2.drawContours(img, [res], 0, (0, 255, 0), 2)
        cv2.drawContours(img, [hull], 0, (0, 0, 255), 3)
    return img, res


def count_fingers(img):
    """
    This function counts the fingers using several methods explained in the presentation.
    """
    thresh = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return 0

    segmented = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(segmented)

    # calculate the extreme points of the hull
    extreme_top = tuple(hull[hull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(hull[hull[:, :, 1].argmax()][0])
    extreme_left = tuple(hull[hull[:, :, 0].argmin()][0])
    extreme_right = tuple(hull[hull[:, :, 0].argmax()][0])

    # calculate the center of the hull
    c_x = int((extreme_left[0] + extreme_right[0]) / 2)
    c_y = int((extreme_top[1] + extreme_bottom[1]) / 2)

    # get the maximum euclidean distance between the center and the extreme points
    distance = pairwise.euclidean_distances([(c_x, c_y)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    maximum_distance = distance[distance.argmax()]

    # calculate the radius using the maximum euclidean distance
    radius = int(0.8 * maximum_distance)

    # draw an auxiliary circle
    circumference = (2 * np.pi * radius)
    circular_roi = np.zeros(thresh.shape[:2], dtype="uint8")
    cv2.circle(circular_roi, (c_x, c_y), radius, 255, 1)

    # now subtract the hand segment off the circle and count the contours left
    circular_roi = cv2.bitwise_and(thresh, thresh, mask=circular_roi)
    cnts, _ = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    count = 0
    # count the contours left
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if ((c_y + (c_y * 0.28)) > (y + h)) and ((circumference * 0.28) > c.shape[0]):
            count += 1

    return max(min(count, 5), 0)


# Main loop
while camera.isOpened():
    ret, frame = camera.read()
    height, width, _ = frame.shape

    # draw a rectangle for the user
    cv2.rectangle(frame, (int(REGION_X * width), 0), (width, int(REGION_Y * height)), (255, 0, 0), 2)

    # place a text at the edge of the window
    original_image = frame.copy()
    cv2.putText(original_image, "Original", (10, 50), font, fontScale, fontColor, lineType, cv2.LINE_AA)

    # calculate the region of interest
    roi = crop_img(frame)
    roi_height, roi_width, _ = roi.shape

    # resize the original image
    original_image = cv2.resize(original_image, (roi_width, roi_height))

    if bg_model is not None:
        roi = remove_background(roi)
        parsed_img = parse_img(roi)

        # count the fingers
        finger_count = count_fingers(parsed_img.copy())
        roi, _ = draw_convex_hull(parsed_img.copy())
        if finger_count is not None:
            cv2.putText(roi, "Finger: "+str(finger_count), (10, 50), font, fontScale, fontColor, lineType, cv2.LINE_AA)

    # draw all pictures
    cv2.imshow(PROGRAM_TITLE, np.hstack((original_image, roi)))

    # check if a key was pressed
    parse_key_command()

    if FIRST_ITERATION:
        FIRST_ITERATION = False
    else:
        # enlarge the window
        cv2.resizeWindow(PROGRAM_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT)
