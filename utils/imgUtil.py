import os
import random
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from skimage.transform import resize

def reshapeAndPad(img, dim):
    """
    Reshapes image with unchanged aspect ratio with padding \n

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

    genColor = lambda: random.randint(0, 255)
    color = (genColor(), genColor(), genColor())

    label = "{0}".format(classes[cls])
    cv2.rectangle(img, topLeft, botRight, color, 1)

    txtSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    botRight = topLeft[0] + txtSize[0] + 3, topLeft[1] + txtSize[1] + 4
    cv2.rectangle(img, topLeft, botRight, color, -1)
    cv2.putText(img, label, (topLeft[0], topLeft[1] + txtSize[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)
    return img


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416):

        with open(list_path, 'r') as file:
            self.img_files = file.readlines()

        self.label_files = [path.replace('Images', 'Labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        self.img_shape = (img_size, img_size)
        self.max_objects = 50

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = np.array(Image.open(img_path))

        # Handles images with less than three channels
        while len(img.shape) != 3:
            index += 1
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path))

        h, w, _ = img.shape
        dim_diff = np.abs(h - w)

        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2

        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))

        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
        padded_h, padded_w, _ = input_img.shape

        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')

        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))

        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        labels = None
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

            # Calculate ratios from coordinates
            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] *= w / padded_w
            labels[:, 4] *= h / padded_h

        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]

        filled_labels = torch.from_numpy(filled_labels)

        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)


if __name__ == '__main__':
    set = ListDataset("../BBox-Label-Tool/trainPath.txt")
    # for x in set:
    #     print(x[1].shape)
    #
    # print(np.transpose(set[0][1].numpy()).shape)

    frame = cv2.imread("../BBox-Label-Tool/Train/Images/000.jpg")
    preppedImg, origImg, dim = prepImage(frame, 416)
    paddedImg, origImg, dim = prepPad(frame, 416)

    print(set[0][1].numpy().shape)
    print(paddedImg[0].numpy().shape)

    cv2.imshow("listdata", np.transpose(set[0][1].numpy(), (1,2,0)))
    cv2.imshow("padded", np.transpose(paddedImg[0].numpy(), (1,2,0)))

    cv2.waitKey(0)
