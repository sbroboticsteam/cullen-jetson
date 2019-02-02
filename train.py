from __future__ import division

import argparse

import torch

from darknet import Darknet
from trainUtil import initWeightsNormal
from utils.txtUtil import loadClasses
from utils.txtUtil import parse_data
from utils.imgUtil import ListDataset
from torch.autograd import Variable

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


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
                        default="160", type=str)
    parser.add_argument("--checkpoint_interval", type=int, help="interval between saving model weights",
                        default=1)
    parser.add_argument("--checkpoint_dir", type=str, help="directory where model checkpoints are saved",
                        default="checkpoints")
    parser.add_argument("--use_cuda", type=bool, help="whether to use cuda if available",
                        default=True)
    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parse()
    dataArgs = parse_data("data/tennisball.data")

    confidence = float(args.confidence)
    nmsThresh = float(args.nms_thresh)
    start = 0
    nCPU = 1
    epochs = 30
    checkpointInterval = 1
    CUDA = torch.cuda.is_available() and args.use_cuda

    paths = {
        "train" : "BBox-Label-Tool/trainPath.txt",
        "val" : "BBox-Label-Tool/valPath.txt"
    }

    # FIXME: Change this to match number of classes in names file
    #  AND change the network's yolo layers to match
    args.cfgfile = "cfg/yolov3-SBRT.cfg"
    numClasses = dataArgs["classes"]
    classes = loadClasses(dataArgs["names"])

    model = Darknet(args.cfgfile)
    model.apply(initWeightsNormal)
    # model.loadWeights(args.weightsfile)

    inpDim = int(args.reso)
    model.netInfo["height"] = args.reso

    # If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()

    # Set the model in training mode. Allows for back prop
    model.train()

    # Training phase
    # Get dataloader
    dataloader = torch.utils.data.DataLoader(
        ListDataset(paths["train"]), batch_size=int(model.netInfo["batch"]), shuffle=False, num_workers=nCPU
    )

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    for epoch in range(epochs):
        for batch_i, (_, imgs, labels) in enumerate(dataloader):
            imgs = Variable(imgs.type(Tensor))
            labels = Variable(labels.type(Tensor), requires_grad=False)

            optimizer.zero_grad()

            loss = model(imgs, labels)


#
#     # Prepare the image as a torch tensor with correct input dimensions
#
#     preppedImg, origImg, dim = prepImage(frame, inpDim)
#
#     # Keep track of original dimensions so we can remove the padding at the end
#     origDim = torch.FloatTensor(dim).repeat(1, 2)
#
#     if CUDA:
#         origDim = origDim.cuda()
#         preppedImg = preppedImg.cuda()
#
#     # Perform forward prop and get output bounding boxes
#     output = model(Variable(preppedImg))
#     output = findTrueDet(output, confidence, numClasses, nmsThresh)
#
#     if type(output) == int:
#         frames += 1
#         print("FPS: {:5.2f}".format(frames / (time.time() - start)))
#
#         cv2.imshow("Frame", frame)
#
#         keyPressed = cv2.waitKey(1)
#         if keyPressed == ord("q"):
#             zed.releaseCam()
#             break
#         continue
#
#     output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(inpDim)) / inpDim
#
#     # Scale up the x1, y1, x2, y2 coordinates to the original image's size
#     output[:, [1, 3]] *= frame.shape[1]
#     output[:, [2, 4]] *= frame.shape[0]
#
#     list(map(lambda x: drawBBoxes(x, classes, frame), output))
#
#     cv2.imshow("Frame", frame)
#     keyPressed = cv2.waitKey(1)
#     if keyPressed == ord("q"):
#         zed.releaseCam()
#         break
#
#     frames += 1
#     print("FPS: {:5.2f}".format(frames / (time.time() - start)))
#
# else:
#     break
