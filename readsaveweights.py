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
    data = parse_data("data/coco.data")
    CUDA = torch.cuda.is_available() and data["use_cuda"]
    device = torch.device("cuda" if CUDA else "cpu")

    confidence = float(data["confidence"])
    nmsThresh = float(data["nms_thresh"])
    # FIXME: Change this to match number of classes in names file
    #  AND change the network's yolo layers to match
    numClasses = 1
    classes = loadClasses(data["names"])

    model = Darknet(data["cfg"])
    # model.loadStateDict("weights/yolov3-320.pt")
    model.loadWeight(data["weights"])
    model.saveStateDict("weights/yolov3-320.pt")

    # model.saveWeights("test.weights")
    # torch.save(model.state_dict(), "weights/yolov3-320.pt")

    # stateDict = torch.load("weights/yolov3-320.pt")
    #
    # for k, v in stateDict.items():
    #     print(k)
