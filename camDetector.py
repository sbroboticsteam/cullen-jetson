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
from zedstreamer import ZedCamera


def arg_parse():
    """
    Parse arguments to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions",
                        default=0.25)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold",
                        default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile",
                        default="weights/yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)
    parser.add_argument("--use_cuda", type=bool, help="whether to use cuda if available",
                        default=True)

    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parse()
    confidence = float(args.confidence)
    nmsThresh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available() and args.use_cuda

    # FIXME: Change this to match number of classes in names file
    #  AND change the network's yolo layers to match
    numClasses = 80
    classes = loadClasses("names/coco.names")

    model = Darknet(args.cfgfile)
    model.loadWeights(args.weightsfile)

    model.netInfo["height"] = args.reso
    inpDim = int(model.netInfo["height"])

    assert inpDim % 32 == 0
    assert inpDim > 32

    # If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()

    # Set the model in evaluation mode. Notifies network to not train
    model.eval()

    # Detection phase
    zed = ZedCamera()
    frames = 0
    start = time.time()

    while True:
        # Get image from camera
        frame = zed.getImage("left")
        frame = np.array(frame[:, :, :3])

        if frame is not None:
            # Prepare the image as a torch tensor with correct input dimensions

            # FIXME: dim is never used
            preppedImg, origImg, dim = prepImage(frame, inpDim)
            print(preppedImg.shape)

            # Keep track of original dimensions so we can remove the padding at the end
            # origDim = torch.FloatTensor(dim).repeat(1, 2)

            if CUDA:
                # origDim = origDim.cuda()
                preppedImg = preppedImg.cuda()

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
            print("FPS: {:5.2f}".format(frames / (time.time() - start)))

        else:
            break
