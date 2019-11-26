import cv2
import imutils


# global variables
background = None

class ImageParser:
    def __init__(self, top, right, bottom, left):
        self.top = top
        self.right = right
        self.bottom = bottom
        self.left = left

        # weight of each image
        self.aWeight = 0.5

        self.num_frames = 0
        self.min_frames = 30


    def find_background(self, image, aWeight):
        """
        This function calculates the weighted sum of the input image.
        :param image: the image to by calculated
        :param aWeight: weight of the input image
        """
        global background

        if background is None:
            background = image.copy().astype("float")
            return

        cv2.accumulateWeighted(image, background, aWeight)

    def segment(self, image, threshold=25):
        """
        Segment the region of the hand within the image
        :param image:
        :param threshold:
        :return:
        """
        global background

        # calculate the difference between the hand and the background
        diff = cv2.absdiff(background.astype("uint8"), image)

        # threshold
        thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

        # contours
        (contours, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # return None, if no contours detected
        if len(contours) == 0:
            return
        else:
            segmented = max(contours, key=cv2.contourArea)
            return (thresholded, segmented)

    def parse(self, frame):

        # Resize
        frame = imutils.resize(frame)

        # Flip
        frame = cv2.flip(frame, 1)

        # Clone
        clone = frame.copy()

        # Get height and width
        (height, width) = frame.shape[:2]

        roi = frame[self.top:self.bottom, self.right:self.left]

        # Grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Blur
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        img = None
        self.num_frames += 1

        if self.num_frames < self.min_frames:
            self.find_background(gray, self.aWeight)
        else:
            # Segmentation
            hand = self.segment(gray)

            if hand is not None:
                (thresholded, segmented) = hand
                return thresholded
                #cv2.drawContours(clone, [segmented + (self.right, self.top)], -1, (0, 0, 255))
                #cv2.imshow("Thesholded", thresholded)

        # Show the hand
        #cv2.rectangle(clone, (self.left, self.top), (self.right, self.bottom), (0, 255, 0), 2)




