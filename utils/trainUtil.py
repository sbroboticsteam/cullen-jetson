import os

import torch
from PIL import Image
from skimage.transform import resize
from torch.utils.data import Dataset
from utils.modelUtil import getIOU
import numpy as np
import math

def initWeightsNormal(module):
    """
    Sets the module's weights to a random number in a normal distribution

    :param module: a layer in the network
    """
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(module.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(module.bias.data, 0.0)


def buildLabels(predBoxes, predConf, predClass, labels, scaledAnch, nA, numCls, gS, ignoreThresh):
    """
    Draws information from the labels tensor, reformats it, and outputs a series of tensors
    which will be used to calculate the loss of the network

    :return:
    """

    # TODO: Convert ground truth information into same format as output
    # Convert targets to padded form

    bS = labels.shape[0]  # batchSize
    numLbls = labels.shape[1]

    trueAMask = torch.zeros(bS, nA, gS, gS)
    invAMask = torch.ones(bS, nA, gS, gS)
    subY = torch.zeros(bS, nA, gS, gS)
    subW = torch.zeros(bS, nA, gS, gS)
    subX = torch.zeros(bS, nA, gS, gS)
    subH = torch.zeros(bS, nA, gS, gS)
    tConf = torch.ByteTensor(bS, nA, gS, gS).fill_(0)
    tClass = torch.ByteTensor(bS, nA, gS, gS, numCls).fill_(0)

    nGT = 0  # Number of ground truth
    nCorrect = 0  # Number of correct predictions

    # For every image
    for b in range(bS):
        # Go through every label
        for t in range(numLbls):

            # If the label doesn't exist ignore it
            if labels[b, t].sum() == 0:
                continue

            nGT += 1

            # We can not use flattenPredict here because its expected format is vastly different than what we have
            # Instead we'll simply perform the same processes as the predict half of flattenPredict

            # ----------------------------------------------------------------
            #  Reformat ground truth boxes so we can compare them to predicted boxes
            # ----------------------------------------------------------------------

            # g = grid = coordinates relative to grid location
            # Convert to position relative to box
            gx = labels[b, t, 1] * gS
            gy = labels[b, t, 2] * gS
            gw = labels[b, t, 3] * gS
            gh = labels[b, t, 4] * gS

            # Get grid indices
            gi = int(gx)  # grid index i
            gj = int(gy)  # grid index j

            # Get shape of ground truth box
            # We don't need its x and y yet since we're trying to figure out which anchor best matches the ground truth box first
            # We can then use this "true" anchor to get loss for the network
            gtBox = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)

            # Combine anchors together so we can get IOU between anchors and ground truth
            anchorShapes = torch.FloatTensor(np.concatenate((np.zeros((len(scaledAnch), 2)), np.array(scaledAnch)), 1))

            # Get the IOUs of the anchor boxes and ground truth box
            anchIOUs = getIOU(gtBox, anchorShapes)

            # FIXME: Honestly not sure what this is for
            # Find which anchors fit most with the ground truth box ??
            invAMask[b, anchIOUs > ignoreThresh, gj, gi] = 0

            # Find index of best matching anchor box
            bestAInd = torch.argmax(anchIOUs)

            # Get the full ground truth box to prep for IOU b/w this and best prediction box
            gtBox = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)

            # FIXME: I think output is not formatted this way
            # We assume the best anchor index is also the best prediction in the outputs
            # If it isn't that means the network is wrong (since the best anchor index is the ground truth anchor)
            # We can get loss out of this
            predBox = predBoxes[b, bestAInd, gj, gi].unsqueeze(0)

            # trueAnchsMask masks the grid points where the ground truth anchor lays
            # invMask is simply the inverse of the trueAnchsMask. We use this to get the false grids later for loss
            trueAMask[b, bestAInd, gj, gi] = 1
            invAMask[b, bestAInd, gj, gi] = 1

            # Records the "subpixel" locations of the center of the ground truth box
            subX[b, bestAInd, gj, gi] = gx - gi
            subY[b, bestAInd, gj, gi] = gy - gj
            subW[b, bestAInd, gj, gi] = math.log(gw / scaledAnch[bestAInd][0] + 1e-16)
            subH[b, bestAInd, gj, gi] = math.log(gh / scaledAnch[bestAInd][1] + 1e-16)

            # One-hot encode labels
            target_label = int(labels[b, t, 0])
            tClass[b, bestAInd, gj, gi, target_label] = 1  # Target's class
            tConf[b, bestAInd, gj, gi] = 1

            # Calculate iou between ground truth and best matching prediction
            iou = getIOU(gtBox, predBox, x1y1x2y2=False)
            predLbl = torch.argmax(predClass[b, bestAInd, gj, gi])
            score = predConf[b, bestAInd, gj, gi]

            if iou > 0.5 and predLbl == target_label and score > 0.5:
                nCorrect += 1

    return nGT, nCorrect, trueAMask, invAMask, subX, subY, subW, subH, tConf, tClass