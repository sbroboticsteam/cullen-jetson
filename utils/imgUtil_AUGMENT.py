import os
import random
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from skimage.transform import resize
import augmentUtil as aug


def reshapeAndPad(img, dim):
    """
    Reshapes image with unchanged aspect ratio with padding.
    Usually called by the prep methods

    :param img: Image to be reshaped
    :type img: numpy.ndarray
    :param dim: Dimensions to reshape to
    :type dim: Tuple
    :return: Reshaped image
    :rtype numpy.ndarray
    """

    imgW, imgH = img.shape[1], img.shape[0]
    w, h = dim

    newW = int(imgW * min(w / imgW, h / imgH))
    newH = int(imgH * min(w / imgW, h / imgH))

    resizedImg = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((dim[1], dim[0], 3), 128)
    canvas[(h - newH) // 2: (h - newH) // 2 + newH,
    (w - newW) // 2: (w - newW) // 2 + newW, :] = resizedImg

    return canvas.astype(np.uint8)


def prepPad(img, inpDim):
    """
    Same as @prepImage but pads the image to ensure aspect ratio is the same when reshaping

    """

    origImg = img
    origDim = origImg.shape[1], origImg.shape[0]
    img = reshapeAndPad(origImg, (inpDim, inpDim))
    preppedImg = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    preppedImg = torch.from_numpy(preppedImg).float().div(255.0).unsqueeze(0)
    return preppedImg, origImg, origDim


def prepImage(img, inpDim):
    """
    Prepares the image as a torch tensor and reshapes it to fit into the network

    :param img: Original image
    :type img: numpy.ndarray
    :param inpDim: Network dimensions to reshape to
    :type inpDim: int
    :return: Tensor-ed image, the original image, the original image's dimensions
    :rtype preppedImg: torch.Tensor
    :rtype origImg: np.ndarray
    :rtype origDim: (int, int)
    """

    origImg = img
    origDim = origImg.shape[1], origImg.shape[0]
    img = cv2.resize(origImg, (inpDim, inpDim))
    preppedImg = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    preppedImg = torch.from_numpy(preppedImg).float().div(255.0).unsqueeze(0)
    return preppedImg, origImg, origDim


def drawBBoxes(dets, classes, img):
    # TODO: Fix docs with proper typing
    """
    Draw bounding boxes of detected objects for visualization purposes

    :param dets: Array of detected objects formatted [batch id, x1, y1, x2, y2, objectness score, class confidence, class]
    :type dets:
    :param classes: List of class names
    :type classes:
    :param img: Image to draw boxes onto
    :type
    :return: Returns image with drawn boxes
    :rtype
    """

    topLeft = tuple(dets[1:3].int())
    botRight = tuple(dets[3:5].int())
    cls = int(dets[-1])
    # print(dets[6])

    genColor = lambda: random.randint(0, 255)
    color = (genColor(), genColor(), genColor())

    label = "{0}".format(classes[cls])
    cv2.rectangle(img, topLeft, botRight, color, 1)

    txtSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    botRight = topLeft[0] + txtSize[0] + 3, topLeft[1] + txtSize[1] + 16
    cv2.rectangle(img, topLeft, botRight, color, -1)
    cv2.putText(img, label, (topLeft[0], topLeft[1] + txtSize[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)
    cv2.putText(img, str(round(float(dets[6]), 2)), (topLeft[0], topLeft[1] + txtSize[1] + 16), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)
    return img


def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=False, multiscale=False):

        with open(list_path, 'r') as file:
            self.img_files = file.readlines()

        self.label_files = [path.replace('Images', 'Labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        self.imgShape = (img_size, img_size)
        self.maxObjects = 50  # Max objects per label txt allowed

        self.augment = augment
        self.multiscale = multiscale
        self.augmentHSV = True
        self.lrFlip = True
        self.udFlip = True

        # SV augmentation by 50%
        self.fraction = 0.50

    def __getitem__(self, index):

        # -------------------- Load Images In --------------------
        imgPath = self.img_files[index % len(self.img_files)].rstrip()
        img = cv2.imread(imgPath)

        # Handles images with less than three channels
        while len(img.shape) != 3:
            index += 1
            imgPath = self.img_files[index % len(self.img_files)].rstrip()
            img = cv2.imread(imgPath)
        # --------------------------------------------------------

        # Augment HSV
        # We do this before everything because padding adds in gray borders
        if self.augmentHSV and self.augment:
            img = aug.randomHSV(img, self.fraction)

        # Set random height if in Multi-Scale training
        if self.multiscale:
            height = random.choice(range(10, 20)) * 32
            self.imgShape = (height, height)

        # ------------- Pad image to preserve aspect ratio during resize -------------
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)

        # Upper (left) and lower (right) padding
        padW, padH = dim_diff // 2, dim_diff - dim_diff // 2

        # Determine padding
        pad = ((padW, padH), (0, 0), (0, 0)) if h <= w else ((0, 0), (padW, padH), (0, 0))

        # Add padding
        inputImg = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        padded_h, padded_w, _ = inputImg.shape
        # ----------------------------------------------------------------------------

        # ------------- Load labels in -------------
        # Reformat cxcywh to xyxy to make augmenting easier
        label_path = self.label_files[index % len(self.img_files)].rstrip()

        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)

            # Extract coordinates for unpadded + unscaled image
            x1 = w * (labels[:, 1] - labels[:, 3] / 2)
            y1 = h * (labels[:, 2] - labels[:, 4] / 2)
            x2 = w * (labels[:, 1] + labels[:, 3] / 2)
            y2 = h * (labels[:, 2] + labels[:, 4] / 2)

            # Adjust for added padding
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]

            # labels[:, 1] = ((x1 + x2) / 2) / padded_w
            # labels[:, 2] = ((y1 + y2) / 2) / padded_h
            # labels[:, 3] *= w / padded_w
            # labels[:, 4] *= h / padded_h

            labels[:, 1] = x1 / padded_w
            labels[:, 2] = y1 / padded_h
            labels[:, 3] = x2 / padded_w
            labels[:, 4] = y2 / padded_h

        else:
            labels = np.array([])

        nL = len(labels)
        # -------------------------------------------

        # ------------- Augment image and do final reformatting -------------
        # Resize image to final shape and un-normalize since skimage auto-normalizes
        inputImg = resize(inputImg, (*self.imgShape, 3), mode='reflect') * 255.

        if self.augment:
            inputImg, labels, M = aug.randomAffine(inputImg, labels, degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.90, 1.10))

        # -------------------------------------------------------------------

        if nL > 0:
            # Change labels from xyxy format back into cxcywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5].copy())

            if self.augment:
                inputImg, labels = aug.randomFlip(inputImg, labels, self.lrFlip, self.udFlip)

        # FIXME: I feel like self.maxObjects is redundant
        # Fill matrix
        filledLabels = np.zeros((self.maxObjects, 5))
        if nL > 0:
            filledLabels[range(len(labels))[:self.maxObjects]] = labels[:self.maxObjects]

        # Channels-first
        inputImg = np.transpose(inputImg, (2, 0, 1))

        inputImg = np.ascontiguousarray(inputImg, np.float64) / 255.

        # As pytorch tensor
        inputImg = torch.from_numpy(inputImg)

        filledLabels = torch.from_numpy(filledLabels)

        return imgPath, inputImg, filledLabels

    def __len__(self):
        return len(self.img_files)

    # def augmentImg(self, img, labels):
    #
    #     img = img.astype(np.uint8)
    #
    #     # Augment affine
    #     img, labels, M = aug.randomAffine(img, labels, degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.90, 1.10))
    #
    #     # Augment Flip
    #     img, labels = aug.randomFlip(img, labels, self.lrFlip, self.udFlip)
    #
    #     img = img.astype(np.float64)
    #     return img, labels


if __name__ == '__main__':
    set = ListDataset("../BBox-Label-Tool/trainPath.txt", augment=True, multiscale=True)

    for _, img, labels in set:
        cv2.imshow("frame", (np.transpose(img.numpy(), (1, 2, 0)) * 255.0).astype(np.uint8))
        cv2.waitKey(0)
