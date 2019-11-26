import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
from image_parser import *

class App:
    def __init__(self, window, window_title, video_source=0):
        self.top, self.right, self.bottom, self.left = 10, 500, 225, 1290
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.gap = 10

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)
        width = int(self.vid.width)
        height = int(self.vid.height)
        self.left = width - self.gap
        self.top = 0 + self.gap
        self.bottom = height - self.gap
        self.right = int(width / 2)

        # image parser
        self.parser = ImageParser(self.top, self.right, self.bottom, self.left)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width=self.vid.width/2, height=self.vid.height/2)
        self.canvas.pack()

        # self.parsedCanvas = tkinter.Canvas(window, width=self.vid.width/2, height=self.vid.height/2)
        # self.parsedCanvas.pack()

        # Button that lets the user take a snapshot
        self.btn_snapshot = tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()
        self.window.mainloop()

    def snapshot(self):
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def update(self):
        ret, frame = self.vid.get_frame()

        if ret:
            frame = cv2.flip(frame, 1)

            # Get height and width
            cv2.rectangle(frame, (self.left, self.top), (self.right, self.bottom), (0, 255, 0), 2)

            img = PIL.Image.fromarray(frame)
            img = img.resize((int(img.width/2), int(img.height/2)), PIL.Image.ANTIALIAS)
            self.photo = PIL.ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

            (grabbed, frame) = self.vid.get_cv2_frame()
            threshold = self.parser.parse(frame)
            if threshold is not None:
                cv2.imshow("Threshold", threshold)
                #threshold = cv2.cvtColor(threshold, cv2.COLOR_BGR2RGB)
                #threshold = PIL.Image.fromarray(threshold)
                #threshold = threshold.resize((int(img.width/2), int(img.height/2)), PIL.Image.ANTIALIAS)
                #trans = PIL.ImageTk.PhotoImage(image=threshold)
                #self.canvas.create_image(0, 0, image=trans, anchor=tkinter.NW)

        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (None, None)

    def get_cv2_frame(self):
        if self.vid.isOpened():
            return self.vid.read()

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


App(tkinter.Tk(), "Tkinter and OpenCV")