import torch.nn as nn

from utils.modelUtil import flattenPredict
from utils.txtUtil import parse_cfg
from utils.modelUtil import getIOU
import numpy as np
import torch

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
        self.ignore_thresh = 0.5

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

        x = x.data
        self.CUDA = x.is_cuda
        output = flattenPredict(x, self.inpDim, self.anchors, self.numClasses, x.is_cuda)

        if trainLabels is not None:
            if self.CUDA:
                self.mseLoss = self.mseLoss.cuda()
                self.bceLoss = self.bceLoss.cuda()
                self.ceLoss = self.ceLoss.cuda()

            # TODO: Convert ground truth information into same format as output
            # Convert targets to padded form

            batchSize, numLabels, numAttrs = trainLabels.shape

            for b in range(batchSize):
                imgLabels = trainLabels[b]
                nonZeroInd = torch.unique(torch.nonzero(imgLabels)[:, 0])
                nonZeroLabels = imgLabels[nonZeroInd]

                # We can not use flattenPredict here because its expected format is vastly different than what we have
                # Instead we'll simply perform the same processes as the predict half of flattenPredict

                stride = self.inpDim // x.size(2)
                gridSize = self.inpDim // stride
                bboxAttrs = 5 + self.numClasses
                numAnchors = len(self.anchors)

                conf_mask = torch.ones(batchSize, numAnchors, gridSize, gridSize)
                bestAnchMask = torch.zeros(batchSize, numAnchors, gridSize, gridSize)
                # tx = torch.zeros(nB, nA, nG, nG)
                # ty = torch.zeros(nB, nA, nG, nG)
                # tw = torch.zeros(nB, nA, nG, nG)
                # th = torch.zeros(nB, nA, nG, nG)
                # tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0)
                # tcls = torch.ByteTensor(nB, nA, nG, nG, nC).fill_(0)

                # ----------------------------------------------------------------
                #  Reformat ground truth boxes so we can compare them to predicted boxes
                # ----------------------------------------------------------------------

                # g = grid = coordinates relative to grid location
                # Convert to position relative to box
                gx = nonZeroLabels[:, 1] * gridSize
                gy = nonZeroLabels[:, 2] * gridSize
                gw = nonZeroLabels[:, 3] * gridSize
                gh = nonZeroLabels[:, 4] * gridSize

                # Scale anchors down to dimension we are working with
                scaledAnchors = [(a[0] / stride, a[1] / stride) for a in self.anchors]

                # Get grid indices
                gridi = int(gx)
                gridj = int(gy)

                # Get shape of ground truth box
                # We don't need its x and y yet since we're trying to figure out which anchor best matches the ground truth box first
                # We can then use this "true" anchor to get loss for the network
                gtBox = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)

                # Combine anchors together so we can get IOU between anchors and ground truth
                anchorShapes = torch.FloatTensor(np.concatenate((np.zeros((len(scaledAnchors), 2)), np.array(scaledAnchors)), 1))

                # Get the IOUs of the anchor boxes and ground truth box
                anchIOUs = getIOU(gtBox, anchorShapes)

                # FIXME: Honestly not sure what this is for
                # Find which anchors fit most with the ground truth box
                anchMask = anchIOUs[b, anchIOUs > self.ignore_thres, gridj, gridi] = 0

                # Find index of best matching anchor box
                bestAnchInd = torch.argmax(anchIOUs)

                # Get the full ground truth box to prep for IOU b/w this and best prediction box
                gtBox = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)

                # FIXME: I think output is not formatted this way
                # We assume the best anchor index is also the best prediction in the outputs
                # If it isn't that means the network is wrong (since the best anchor index is the ground truth anchor)
                # We can get loss out of this
                predBox = output[b, bestAnchInd, gridj, gridi]

                # Mask the points which the best anchor encompasses
                bestAnchMask[b, bestAnchInd, gridj, gridi] = 1

                # FIXME Remove this at the end of testing and replace with proper return
                return output, None


        else:
            return output


class Darknet(nn.Module):
    """
    Class containing the entire YOLO network
    """

    def __init__(self, cfgPath):
        """
        :param cfgPath: path to cfg file
        :type cfgPath: str
        """
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgPath)
        self.netInfo, self.modulesList = createModules(self.blocks)

    def forward(self, x, trainLabels=None):
        isTraining = trainLabels is not None

        modules = self.blocks[1:]

        outputs = {}  # Keep track of previous outputs for route and shortcut layers. Layer index : Feature map
        out = []  # The final output from the network which consists of a tensor of bbox attributes

        haveDet = False
        for layerInd, module in enumerate(modules):
            moduleType = module["type"]

            # If the module is just a conv or upsample layer, forward pass it
            if moduleType in {"convolutional", "upsample", "maxpool"}:
                x = self.modulesList[layerInd](x)
                outputs[layerInd] = x

            elif moduleType == "route":

                layers = module["layers"]  # Get the layers we are routing
                layers = [int(a) for a in layers]

                if layers[0] > 0:
                    layers[0] = layers[0] - layerInd

                # If the route layer only has one parameter, we just get the output of that layer
                if len(layers) == 1:
                    x = outputs[layerInd + layers[0]]
                # If the route layer has two parameters, concatenate the layers accordingly
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - layerInd

                    featMap1 = outputs[layerInd + layers[0]]
                    featMap2 = outputs[layerInd + layers[1]]

                    x = torch.cat((featMap1, featMap2), 1)

                outputs[layerInd] = x

            elif moduleType == "shortcut":
                from_ = int(module["from"])
                x = outputs[layerInd - 1] + outputs[layerInd + from_]

                outputs[layerInd] = x

            elif moduleType == "yolo":

                # FIXME: This x.data may not be needed
                x = x.data
                if isTraining:
                    # TODO: Get losses from layer and process them into field var
                    x, losses = self.modulesList[layerInd][0](x, trainLabels)  # I need to access 0th index so I can get DetectionLayer exactly. Otherwise it throws positional args error


                else:

                    x = self.modulesList[layerInd][0](x)
                    if not haveDet:
                        out = x
                        haveDet = True
                    else:
                        out = torch.cat((out, x), 1)

                outputs[layerInd] = outputs[layerInd - 1]

        # out can have two different things inside it
        # One is a tensor of all object detections and their attributes, if the model is in validation mode
        # The second is the total loss of the network, if the model is in training mode
        if isTraining:
            return sum(out)
        else:
            return out

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
        self.header_info.tofile(wf)

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
    prevFilters = 3  # The initial layer accepts the raw image, so its depth is 3 (RGB)
    outFilters = []

    for layerNum, block in enumerate(blocks[1:]):
        module = nn.Sequential()

        # Check what type of layer we are working with
        # In total we have: convolutional, upsample, route, shortcut
        if block["type"] == "convolutional":

            # ----------------------------------------
            #  Get all the information about the layer
            # ----------------------------------------

            activation = block["activation"]
            try:
                batchNorm = int(block["batch_normalize"])
                bias = False
            except:
                batchNorm = 0
                bias = True

            filters = int(block["filters"])
            kernelSize = int(block["size"])
            stride = int(block["stride"])
            pad = (kernelSize - 1) // 2 if int(block["pad"]) else 0

            # -------------------------------
            #  Add each layer to modules list
            # -------------------------------

            # Convolutional layer
            conv = nn.Conv2d(prevFilters, filters, kernelSize, stride, pad, bias=bias)
            module.add_module("Convolutional_{0}".format(layerNum), conv)

            # Batch Normalization layer
            if batchNorm:
                bn = nn.BatchNorm2d(filters)
                module.add_module("BatchNormalization_{0}".format(layerNum), bn)

            # Activation layer
            if activation == "leaky":
                activ = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("LeakyReLU_{0}".format(layerNum), activ)

        elif block["type"] == "maxpool":

            kernelSize = int(block["size"])
            stride = int(block["stride"])

            if kernelSize == 2 and stride == 1:
                padding = nn.ZeroPad2d((0, 1, 0, 1))
                module.add_module("_debug_padding_%d" % layerNum, padding)
            else:
                padding = int((kernelSize - 1) // 2)

            maxPool = nn.MaxPool2d(kernelSize, stride, padding)
            module.add_module("MaxPool_{}".format(layerNum), maxPool)

        elif block["type"] == "upsample":
            stride = int(block["stride"])
            upsample = nn.Upsample(scale_factor=stride, mode="nearest")
            module.add_module("Upsample_{}".format(layerNum), upsample)

        # Route layer either returns the feature map of the layer at specified index
        # Or the concatenated feature maps of two layers along the depth dimension
        elif block["type"] == "route":
            block["layers"] = block["layers"].split(",")

            start = int(block["layers"][0])
            try:
                end = int(block["layers"][1])
            except:
                end = 0

            # Positive annotation
            if start > 0:
                start = start - layerNum
            if end > 0:
                end = end - layerNum

            route = EmptyLayer()
            module.add_module("Route_{0}".format(layerNum), route)

            if end < 0:
                filters = outFilters[layerNum + start] + outFilters[layerNum + end]
            else:
                filters = outFilters[layerNum + start]

        # Shortcut layers mean skip connection
        # The output is simply adding the feature maps from the previous and ith layer back
        # i is a parameter of the shortcut layer
        elif block["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("Shortcut_{}".format(layerNum), shortcut)

        elif block["type"] == "yolo":
            # Mask determines which anchors we are using
            mask = block["mask"].split(",")
            mask = [int(m) for m in mask]

            anchors = block["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            numClasses = int(block["classes"])
            inpDim = int(netInfo["height"])

            detection = DetectionLayer(anchors, inpDim, numClasses)
            module.add_module("Detection_{}".format(layerNum), detection)

        else:
            print("--------BLOCK TYPE UNKNOWN--------")
            print("Block: {}".format(block["type"]))

        moduleList.append(module)
        prevFilters = filters
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
