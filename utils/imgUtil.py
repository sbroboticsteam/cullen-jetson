import os
import random
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from skimage.transform import resize
import math
from torchvision import transforms


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


import imgaug.augmenters as iaa
import imgaug.parameters as iap
import imgaug as ia


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416):

        with open(list_path, 'r') as file:
            self.img_files = file.readlines()

        self.label_files = [path.replace('Images', 'Labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        self.img_shape = (img_size, img_size)
        self.max_objects = 50

        augment = True
        multiscale = False
        augmentHSV = True
        lrFlip = True
        udFlip = True

        self.augments = iaa.Sequential([
            iaa.Resize(
                {"height": 320, "width": 320}
            ),

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
                cval=(128, 128),
                mode=ia.ALL
            ),

        ])

    def __getitem__(self, index):
        seqDet = self.augments.to_deterministic()
        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = cv2.imread(img_path)
        # img = np.array(Image.open(img_path))

        # Handles images with less than three channels
        while len(img.shape) != 3:
            index += 1
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = cv2.imread(img_path)
            # img = np.array(Image.open(img_path))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)

        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2

        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))

        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=128)
        padded_h, padded_w, _ = input_img.shape

        # ---------
        #  Label
        # ---------
        cv2.imwrite("test.png", input_img)

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

            # # Calculate ratios from coordinates
            # labels[:, 1] = ((x1 + x2) / 2) / padded_w
            # labels[:, 2] = ((y1 + y2) / 2) / padded_h
            # labels[:, 3] *= w / padded_w
            # labels[:, 4] *= h / padded_h

            labels[:, 1] = x1
            labels[:, 2] = y1
            labels[:, 3] = x2
            labels[:, 4] = y2

        else:
            labels = np.array([])

        bbs = ia.BoundingBoxesOnImage([ia.BoundingBox(x1=label[1], y1=label[2], x2=label[3], y2=label[4]) for label in labels], shape=input_img.shape)
        input_img = seqDet.augment_images([input_img])[0]
        bbsAug = seqDet.augment_bounding_boxes([bbs])[0]
        input_img = np.ascontiguousarray(input_img)

        for i in range(len(bbs.bounding_boxes)):
            before = bbs.bounding_boxes[i]
            after = bbsAug.bounding_boxes[i]
            print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
                i,
                before.x1, before.y1, before.x2, before.y2,
                after.x1, after.y1, after.x2, after.y2)
                  )
        image_before = bbs.draw_on_image(img, thickness=2)
        image_after = bbs.draw_on_image(input_img, thickness=2, color=[0, 0, 255])
        cv2.imshow("test", image_before)
        cv2.imshow("test2", image_after)
        cv2.waitKey(0)

        input_img = input_img / 255.
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        return img_path, input_img, [0]

    def __len__(self):
        return len(self.img_files)


if __name__ == '__main__':
    set = ListDataset("../BBox-Label-Tool/trainPath.txt")

    for _, img, labels in set:
        print(labels)
        cv2.imshow("frame", (np.transpose(img.numpy(), (1, 2, 0)) * 255.0).astype(np.uint8))
        cv2.waitKey(0)
