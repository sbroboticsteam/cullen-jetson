import os
import random
from typing import Tuple

import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import torch
from skimage.transform import resize
from torch.utils.data import Dataset


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


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True):

        with open(list_path, 'r') as file:
            self.img_files = file.readlines()

        self.label_files = [path.replace('Images', 'Labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        self.img_shape = (img_size, img_size)
        self.max_objects = 50

        # augment = True
        # multiscale = False
        # augmentHSV = True
        # lrFlip = True
        # udFlip = True

        if augment:
            self.augments = iaa.Sequential([

                iaa.WithColorspace(
                    to_colorspace="HSV",
                    from_colorspace="RGB",
                    children=[iaa.WithChannels(1, iaa.Add((-50, 50))),
                              iaa.WithChannels(2, iaa.Add((-50, 50)))]
                ),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Affine(
                    scale={"x": (0.90, 1.10), "y": (0.90, 1.10)},
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    rotate=(-10, 10),
                    shear=(-16, 16),
                    order=[0, 1],
                    mode="constant",
                    cval=128
                ),

            ])
        else:
            self.augments = iaa.Noop()

    def __getitem__(self, index):
        seqDet = self.augments.to_deterministic()

        # --------------------------------------------------------------------------------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = cv2.imread(img_path)

        # Handles images with less than three channels
        while len(img.shape) != 3:
            index += 1
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = cv2.imread(img_path)

        h, w, _ = img.shape

        # --------------------------------------------------------------------------------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)

            x1 = w * (labels[:, 1] - labels[:, 3] / 2)
            y1 = h * (labels[:, 2] - labels[:, 4] / 2)
            x2 = w * (labels[:, 1] + labels[:, 3] / 2)
            y2 = h * (labels[:, 2] + labels[:, 4] / 2)

            labels[:, 1] = x1
            labels[:, 2] = y1
            labels[:, 3] = x2
            labels[:, 4] = y2

        else:
            labels = np.array([])

        # --------------------------------------------------------------------------------

        bbs = ia.BoundingBoxesOnImage([ia.BoundingBox(x1=label[1], y1=label[2], x2=label[3], y2=label[4], label=label[0]) for label in labels], shape=img.shape)
        augImg = seqDet.augment_images([img])[0]
        augImg = np.ascontiguousarray(augImg)
        augBBS = seqDet.augment_bounding_boxes([bbs])[0]
        augBBS = augBBS.remove_out_of_image().clip_out_of_image()

        # --------------------------------------------------------------------------------

        dim_diff = np.abs(h - w)

        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2

        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))

        paddedResize = iaa.Sequential([
            iaa.PadToFixedSize(w + (2 * pad[1][0]), h + (2 * pad[0][0]), position="center", pad_mode='constant', pad_cval=128),
            iaa.Resize({"height": self.img_shape[0], "width": self.img_shape[1]})

        ])

        resizedImg = paddedResize.augment_images([augImg])[0]
        resizedBBS = paddedResize.augment_bounding_boxes([augBBS])[0]

        # --------------------------------------------------------------------------------

        # origDebug = bbs.draw_on_image(img, thickness=2, color=[0, 255, 0])
        # augmentedDebug = augBBS.draw_on_image(augImg, thickness=2, color=[0, 0, 255])
        # inputDebug = resizedBBS.draw_on_image(resizedImg, thickness=2, color=[255, 0, 0])
        # print("Original: ", origDebug.shape)
        # print("Augmented: ", augmentedDebug.shape)
        # print("Resized: ", inputDebug.shape)
        # cv2.imshow("orig", origDebug)
        # cv2.imshow("augmented", augmentedDebug)
        # cv2.imshow("resized", inputDebug)
        # cv2.waitKey(0)

        # --------------------------------------------------------------------------------
        resizedH, resizedW, _ = resizedImg.shape

        # Normalize
        inputImg = resizedImg / 255.

        # Channels-first
        inputImg = np.transpose(inputImg, (2, 0, 1))

        # As pytorch tensor
        inputImg = torch.from_numpy(inputImg).float()

        labelsArr = np.zeros((len(resizedBBS.bounding_boxes), 5), dtype=np.float32)
        for i, box in enumerate(resizedBBS.bounding_boxes):
            cx = ((box.x1 + box.x2) / 2) / resizedW
            cy = ((box.y1 + box.y2) / 2) / resizedH
            w = (box.x2 - box.x1) / resizedW
            h = (box.y2 - box.y1) / resizedH

            labelsArr[i] = [box.label, cx, cy, w, h]

        # Fill matrix
        filledLabels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filledLabels[range(len(labelsArr))[:self.max_objects]] = labelsArr[:self.max_objects]

        filledLabels = torch.from_numpy(filledLabels)

        return img_path, inputImg, filledLabels

    def __len__(self):
        return len(self.img_files)


if __name__ == '__main__':
    set = ListDataset("../BBox-Label-Tool/trainPath.txt")

    for _, img, labels in set:
        print(labels)
        cv2.imshow("frame", (np.transpose(img.numpy(), (1, 2, 0)) * 255.0).astype(np.uint8))
        cv2.waitKey(0)
