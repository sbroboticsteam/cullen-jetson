import cv2
from DistAngleVision import Vision
import numpy as np
import sys
import matplotlib.pyplot as plt
from zedstreamer import ZedCamera

if __name__ == '__main__':

    plt.ion()
    axes = plt.gca()
    axes.set_ylim([0, 500])

    # for x in range(0, 5):
    #     stream = cv2.VideoCapture(x)
    #
    #     if (stream.isOpened()):
    #         print("Camera found on port: %d" % (x))
    #         break
    #
    # if (not stream.isOpened()):
    #     print("Camera not found")
    #     sys.exit()
    zed = ZedCamera()

    # FIXME give half horz FOV of zed camera
    vision = Vision(30)

    i = 0
    while True:
        # ret, src = stream.read()
        src = zed.getImage("left")

        if src is not None:

            vision.processImg(src)

            keyPressed = cv2.waitKey(33)
            if keyPressed == ord("s"):
                cv2.imwrite("{}_{}.png".format("debug_output", i), vision.getSourceImg())
                i += 1
            elif keyPressed == ord("q"):
                cv2.destroyAllWindows()
                sys.exit()
