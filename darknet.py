import torch.nn as nn

from utils.modelUtil import flattenPredict
from utils.modelUtil import predict
from utils.trainUtil import buildLabels
from utils.txtUtil import parse_cfg
from utils.modelUtil import getIOU
import numpy as np
import torch
import math
from collections import defaultdict

"""
This module stores all of the layer classes for the network as well as the main network class itself
"""


class EmptyLayer(nn.Module):

    def __init__(self):
        """
        Dummy layer for shortcut blocks
        """

        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):

    def __init__(self, anchors, inpDim, numClasses):
        # FIXME: Fix pydocs for this function
        """
        YOLO's custom detection layer

        :param anchors: list of anchors being used in the detection layer
        :type anchors: list
        """

        super(DetectionLayer, self).__init__()
        self.anchors = anchors
        self.numClasses = numClasses
        self.inpDim = inpDim

        # FIXME: These are hard coded because I don't know how to pass these two parameters efficiently
        #  They are, however, functionally the same as the ones in the main class
        self.confidence = 0.25
        self.nmsThresh = 0.4

        # FIXME: I don't know what this is supposed to represent
        self.ignoreThresh = 0.5

        self.mseLoss = nn.MSELoss(size_average=True)  # Coordinate loss
        self.bceLoss = nn.BCELoss(size_average=True)  # Confidence loss
        self.ceLoss = nn.CrossEntropyLoss()  # Class loss

    def forward(self, x, trainLabels=None):
        # FIXME: Fix pydocs for this function
        """
        :param x: Previous layer's output
        :param inpDim: Dimension of original image input
        :param trainLabels: Tensor of ground truth labels
        :return:
        """
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.CUDA = x.is_cuda

        if trainLabels is not None:

            if self.CUDA:
                self.mseLoss = self.mseLoss.cuda()
                self.bceLoss = self.bceLoss.cuda()
                self.ceLoss = self.ceLoss.cuda()

            # p = prediction
            px, py, pw, ph, predBoxes, predConf, predClass, scaledAnchors = predict(
                x,
                self.inpDim,
                self.anchors,
                self.numClasses,
                x.is_cuda
            )

            # t for target or true
            nGT, nCorrect, mask, confMask, tx, ty, tw, th, tconf, tcls = buildLabels(
                predBoxes=predBoxes.cpu().data,
                predConf=predConf.cpu().data,
                predClass=predClass.cpu().data,
                labels=trainLabels.cpu().data,
                scaledAnch=scaledAnchors.cpu().data,
                nA=len(self.anchors),
                numCls=self.numClasses,
                gS=x.size(2),
                ignoreThresh=self.ignoreThresh,
            )

            nProposals = int((predConf > 0.5).sum().item())
            recall = float(nCorrect / nGT) if nGT else 1
            precision = float(nCorrect / nProposals) if nProposals > 0 else 0

            # Reformat all of the outputs from buildLabels for efficiency(?)
            mask = Variable(mask.type(ByteTensor))
            confMask = Variable(confMask.type(ByteTensor))

            tx = Variable(tx.type(FloatTensor), requires_grad=False)
            ty = Variable(ty.type(FloatTensor), requires_grad=False)
            tw = Variable(tw.type(FloatTensor), requires_grad=False)
            th = Variable(th.type(FloatTensor), requires_grad=False)
            tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
            tcls = Variable(tcls.type(LongTensor), requires_grad=False)

            # Get conf mask where gt and where there is no gt
            confMaskTrue = mask
            confMaskFalse = confMask - mask

            # Mask outputs to ignore non-existing objects
            loss_x = self.mseLoss(px[mask], tx[mask])
            loss_y = self.mseLoss(py[mask], ty[mask])
            loss_w = self.mseLoss(pw[mask], tw[mask])
            loss_h = self.mseLoss(ph[mask], th[mask])
            loss_conf = self.bceLoss(predConf[confMaskFalse],
                                     tconf[confMaskFalse]) + self.bceLoss(predConf[confMaskTrue], tconf[confMaskTrue])

            loss_cls = (1 / x.size(0)) * self.ceLoss(predClass[mask], torch.argmax(tcls[mask], 1))
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            return (
                loss,
                loss_x.item(),
                loss_y.item(),
                loss_w.item(),
                loss_h.item(),
                loss_conf.item(),
                loss_cls.item(),
                recall,
                precision,
            )

        else:
            output = flattenPredict(x, self.inpDim, self.anchors, self.numClasses, x.is_cuda)

            return output


class Darknet(nn.Module):
    """
    Class containing the entire YOLO network
    """

    def __init__(self, cfgPath, feature_extract=False):
        """
        :param cfgPath: path to cfg file
        :type cfgPath: str
        """
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgPath)
        self.netInfo, self.modulesList = createModules(self.blocks)
        self.lossNames = ["x", "y", "w", "h", "conf", "cls", "recall", "precision"]  # FIXME maybe make this the losses dict as well

        self.seen = 0
        self.headerInfo = np.array([0, 0, 0, self.seen, 0])

        if feature_extract:
            self.setReqGrad()

    def forward(self, x, trainLabels=None):
        isTraining = trainLabels is not None

        modules = self.blocks[1:]

        layerOuts = []  # Keep track of previous outputs for route and shortcut layers
        out = []  # The final output from the network which consists of a tensor of bbox attributes
        self.losses = defaultdict(float)  # Losses dictionary to keep track of type of loss and their values

        for layerInd, module in enumerate(modules):
            moduleType = module["type"]

            # If the module is just a conv or upsample layer, forward pass it
            if moduleType in {"convolutional", "upsample", "maxpool"}:
                x = self.modulesList[layerInd](x)

            elif moduleType == "route":
                layer_i = [int(x) for x in module["layers"].split(",")]
                x = torch.cat([layerOuts[i] for i in layer_i], 1)

            elif moduleType == "shortcut":
                from_ = int(module["from"])
                x = layerOuts[-1] + layerOuts[from_]

                # x = prevOuts[layerInd - 1] + prevOuts[layerInd + from_]

                # prevOuts[layerInd] = x

            elif moduleType == "yolo":

                if isTraining:
                    # TODO: Get losses from layer and process them into field var
                    x, *losses = self.modulesList[layerInd][0](x, trainLabels)  # I need to access 0th index so I can get DetectionLayer exactly. Otherwise it throws positional args error

                    for name, loss in zip(self.lossNames, losses):
                        self.losses[name] += loss

                else:
                    x = self.modulesList[layerInd][0](x)

                out.append(x)

            layerOuts.append(x)

        self.losses["recall"] /= 3
        self.losses["precision"] /= 3

        # out can have two different things inside it
        # One is a tensor of all object detections and their attributes, if the model is in validation mode
        # The second is the total loss of the network, if the model is in training mode
        if isTraining:
            return sum(out)
        else:
            return torch.cat(out, 1)

    def loadWeights(self, weightPath):
        """
        Loads weights from a weight file into network

        :param weightPath: Path to weight file
        :type weightPath: str
        """
        wf = open(weightPath, "rb")

        header = np.fromfile(wf, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seenImgs = self.header[3]

        weights = np.fromfile(wf, dtype=np.float32)

        ptr = 0  # Pointer since weight values are in serial
        for i in range(len(self.modulesList)):
            moduleType = self.blocks[i + 1]["type"]  # Remember first block is net block

            if moduleType == "convolutional":
                model = self.modulesList[i]

                try:
                    batchNorm = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batchNorm = 0

                conv = model[0]  # First layer in convolutional block is always the conv layer

                # Load batch norm weights if there are any
                if batchNorm:
                    bn = model[1]  # Second layer in conv block is always the batch norm layer if there is one
                    num_bn_biases = bn.bias.numel()  # Get the number of weights in batch norm layer

                    # Loading weights
                    bnBiases = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bnWeights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bnRunningMean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bnRunningVar = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # Cast weights into proper dimension shape
                    bnBiases = bnBiases.view_as(bn.bias.data)
                    bnWeights = bnWeights.view_as(bn.weight.data)
                    bnRunningMean = bnRunningMean.view_as(bn.running_mean)
                    bnRunningVar = bnRunningVar.view_as(bn.running_var)

                    # Copy data into model
                    bn.bias.data.copy_(bnBiases)
                    bn.weight.data.copy_(bnWeights)
                    bn.running_mean.copy_(bnRunningMean)
                    bn.running_var.copy_(bnRunningVar)

                else:
                    numBiases = conv.bias.numel()

                    convBiases = torch.from_numpy(weights[ptr: ptr + numBiases])
                    ptr += numBiases

                    convBiases = convBiases.view_as(conv.bias.data)

                    conv.bias.data.copy_(convBiases)

                # Load conv weights
                numWeights = conv.weight.numel()

                convWeights = torch.from_numpy(weights[ptr: ptr + numWeights])
                ptr += numWeights

                convWeights = convWeights.view_as(conv.weight.data)
                conv.weight.data.copy_(convWeights)

    def saveWeights(self, savePath):
        """
        Saves the weights from trained model into new weight file

        :param savePath: Path name of new weights file
        """
        wf = open(savePath, "wb")

        # Attach the header at the top of the file
        self.header[3] = self.seen
        self.headerInfo.tofile(wf)

        # Now, let us save the weights
        for i, (block, module) in enumerate(zip(self.blocks[1:], self.modulesList[:])):
            if block["type"] == "convolutional":
                convLayer = module[0]

                try:
                    # This dict access here is to trigger the try except if the key doesn't exist
                    batchNorm = int(block["batch_normalize"])

                    bnLayer = module[1]
                    bnLayer.bias.data.cpu().numpy().tofile(wf)
                    bnLayer.weight.data.cpu().numpy().tofile(wf)
                    bnLayer.running_mean.data.cpu().numpy().tofile(wf)
                    bnLayer.running_var.data.cpu().numpy().tofile(wf)
                except:
                    convLayer.bias.data.cpu().numpy().tofile(wf)

                convLayer.weight.data.cpu().numpy().tofile(wf)

        wf.close()

    def setReqGrad(self):
        # TODO: Add pydocs
        # FIXME: Holy crap the number of iterations needed for something so simple. Pls fix

        for param in self.parameters():
            param.requires_grad = False

        for i, module in enumerate(self.blocks[1:]):
            if module["type"] == "yolo":
                gradInd = str(i - 1)

                for name, param in self.named_parameters():
                    if gradInd in name:
                        param.requires_grad = True


def createModules(blocks):
    """
    Creates a list of formal modules after processing the cfg file. Prep method for creating the PyTorch network

    :param blocks: List of all of the blocks and their configurations
    :type blocks: list

    :return: Tuple containing the network information dict and the modules list
    :rtype: (dict, nn.ModuleList)
    """

    netInfo = blocks[0]  # Remember the 0th block in the cfg file is information about network
    moduleList = nn.ModuleList()
    outFilters = [int(netInfo["channels"])]

    for layerNum, block in enumerate(blocks[1:]):
        modules = nn.Sequential()

        # Check what type of layer we are working with
        # In total we have: convolutional, upsample, route, shortcut
        if block["type"] == "convolutional":

            activation = block["activation"]
            try:
                batchNorm = int(block["batch_normalize"])
            except:
                batchNorm = 0
            bias = not batchNorm

            filters = int(block["filters"])
            kernelSize = int(block["size"])
            stride = int(block["stride"])
            pad = (kernelSize - 1) // 2 if int(block["pad"]) else 0

            # Convolutional layer
            conv = nn.Conv2d(outFilters[-1], filters, kernelSize, stride, pad, bias=bias)
            modules.add_module("Convolutional_{0}".format(layerNum), conv)

            # Batch Normalization layer
            if batchNorm:
                bn = nn.BatchNorm2d(filters)
                modules.add_module("BatchNormalization_{0}".format(layerNum), bn)

            # Activation layer
            if activation == "leaky":
                activ = nn.LeakyReLU(0.1, inplace=True)
                modules.add_module("LeakyReLU_{0}".format(layerNum), activ)

        elif block["type"] == "maxpool":

            kernelSize = int(block["size"])
            stride = int(block["stride"])

            if kernelSize == 2 and stride == 1:
                padding = nn.ZeroPad2d((0, 1, 0, 1))
                modules.add_module("_debug_padding_%d" % layerNum, padding)
            else:
                padding = int((kernelSize - 1) // 2)

            maxPool = nn.MaxPool2d(kernelSize, stride, padding)
            modules.add_module("MaxPool_{}".format(layerNum), maxPool)

        elif block["type"] == "upsample":
            stride = int(block["stride"])
            upsample = nn.Upsample(scale_factor=stride, mode="nearest")
            modules.add_module("Upsample_{}".format(layerNum), upsample)

        # Route layer either returns the feature map of the layer at specified index
        # Or the concatenated feature maps of two layers along the depth dimension
        elif block["type"] == "route":
            layers = [int(x) for x in block["layers"].split(",")]
            filters = sum([outFilters[layer_i] for layer_i in layers])
            modules.add_module("Route_{}".format(layerNum), EmptyLayer())

        # Shortcut layers mean skip connection
        # The output is simply adding the feature maps from the previous and ith layer back
        # i is a parameter of the shortcut layer
        elif block["type"] == "shortcut":
            filters = outFilters[int(block["from"])]
            modules.add_module("Shortcut_{}".format(layerNum), EmptyLayer())

        elif block["type"] == "yolo":
            # Mask determines which anchors we are using
            anchMask = block["mask"].split(",")
            anchMask = [int(m) for m in anchMask]

            anchors = block["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchMask]

            numClasses = int(block["classes"])
            inpDim = int(netInfo["height"])

            detection = DetectionLayer(anchors, inpDim, numClasses)
            modules.add_module("Detection_{}".format(layerNum), detection)

        else:
            print("--------BLOCK TYPE UNKNOWN--------")
            print("Block: {}".format(block["type"]))

        moduleList.append(modules)
        outFilters.append(filters)

    return (netInfo, moduleList)


import cv2
from torch.autograd import Variable
from utils.modelUtil import findTrueDet
from utils.imgUtil import drawBBoxes
from utils.txtUtil import loadClasses
from utils.imgUtil import prepImage

if __name__ == '__main__':
    model = Darknet("cfg/yolov3.cfg")
    model.loadWeights("weights/yolov3.weights")

    classes = loadClasses("names/coco.names")

    img = cv2.imread("dog-cycle-car.png")
    imgT, orig, dim = prepImage(img, 416)
    # img = cv2.resize(img, (416, 416))  # Resize to input dimension of network
    # imgT = img[:, :, ::-1].transpose((2, 0, 1)).copy()  # BGR --> RGB | Height x Width x Channel --> Channel x Height x Width
    # imgT = torch.from_numpy(imgT).float().div(255.0).unsqueeze(0)  # Add a channel at 0 for batch | Normalize
    # imgT = Variable(imgT)

    # if torch.cuda.is_available():
    #     model.cuda()
    #     imgT.cuda()

    output = model(imgT)
    output = findTrueDet(output, 0.25, 80, 0.4)  # batch id???, x1, y1, x2, y2, objectness score, class confidence, class

    # output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(416)) / 416
    #
    # # Scale up the x1, y1, x2, y2 coordinates to the original image's size
    # output[:, [1, 3]] *= img.shape[1]
    # output[:, [2, 4]] *= img.shape[0]
    #
    # list(map(lambda x: drawBBoxes(x, classes, img), output))
    #
    # cv2.imshow("Frame", img)
    # cv2.waitKey(0)
