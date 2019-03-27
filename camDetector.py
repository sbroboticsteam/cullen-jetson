from __future__ import division

import argparse
import time

import cv2
import numpy as np
import torch
from torch.autograd import Variable

from darknet import Darknet
from utils.imgUtil import drawBBoxes
from utils.imgUtil import prepImage
from utils.modelUtil import findTrueDet
from utils.txtUtil import loadClasses
from utils.txtUtil import parse_data
from zedstreamer import ZedCamera




if __name__ == '__main__':
    data = parse_data("data/tennisball_VAL.data")
    CUDA = torch.cuda.is_available() and data["use_cuda"]
    device = torch.device("cuda" if CUDA else "cpu")

    confidence = float(data["confidence"])
    nmsThresh = float(data["nms_thresh"])
    # FIXME: Change this to match number of classes in names file
    #  AND change the network's yolo layers to match
    numClasses = 1
    classes = loadClasses(data["names"])

    model = Darknet(data["cfg"])
    # model.loadWeight(data["weights"])
    model.loadStateDict("checkpoints/epoch_9.pt")

    inpDim = int(data["reso"])

    # If there's a GPU availible, put the model on GPU
    model.to(device)

    # Set the model in evaluation mode. Notifies network to not train
    model.eval()

    # Detection phase
    zed = ZedCamera()
    zed.resetSettings()
    # zed.setCamSettings(brightness=4,
    #                    contrast=0,
    #                    hue=0,
    #                    sat=4,
    #                    gain=70,
    #                    exp=75)
    frames = 0
    start = time.time()

    while True:
        # Get image from camera
        frame = zed.getImage("left")
        # ret, frame = stream.read()

        if frame is not None:
            frame = np.array(frame[:, :, :3])

            # Prepare the image as a torch tensor with correct input dimensions
            # FIXME: dim is never used
            preppedImg, origImg, dim = prepImage(frame, inpDim)

            # Keep track of original dimensions so we can remove the padding at the end
            # origDim = torch.FloatTensor(dim).repeat(1, 2)

            if CUDA:
                # origDim = origDim.cuda()
                preppedImg = preppedImg.cuda()

            with torch.no_grad():
                # Perform forward prop and get output bounding boxes
                output = model(Variable(preppedImg))
                output = findTrueDet(output, confidence, numClasses, nmsThresh)


            if type(output) == int:
                frames += 1
                print("FPS: {:5.2f}".format(frames / (time.time() - start)))

                cv2.imshow("Frame", frame)

                keyPressed = cv2.waitKey(1)
                if keyPressed == ord("q"):
                    zed.releaseCam()
                    break
                continue

            # print("Output: {}".format(output.shape))
            # print(output)

            output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(inpDim)) / inpDim

            # Scale up the x1, y1, x2, y2 coordinates to the original image's size
            output[:, [1, 3]] *= frame.shape[1]  # x1, x2
            output[:, [2, 4]] *= frame.shape[0]  # y1, y2

            list(map(lambda x: drawBBoxes(x, classes, frame), output))

            cv2.imshow("Frame", frame)
            keyPressed = cv2.waitKey(1)
            if keyPressed == ord("q"):
                zed.releaseCam()
                break

            frames += 1
            # print("FPS: {:5.2f}".format(frames / (time.time() - start)))

        else:
            continue
