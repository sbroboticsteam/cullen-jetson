import numpy as np
import torch

"""
    Normally the feature map is a grid laid out as such: [grid_x, grid_y, attrs] (not in code literally)
    where grid_x and grid_y is the cell at position x,y in the grid
    and attrs is a list of many lists of bounding box attributes: [ [x, y, w, h, objScore, c1, c2, ...], ... ]
    Note the total number of attributes lists is equivalent to (5 + numClasses) * 3, 3 being the number of anchors
    
    This method takes that feature map and transforms it into a 2D tensor
    such that each row of the tensor corresponds to the attributes of a bounding box:
    [ [cx, cy, w, h, objScore, c1, c2, ...] for grid[0,0]'s 1st BBox
      [cx, cy, w, h, objScore, c1, c2, ...] for grid[0,0]'s 2nd BBox
      [cx, cy, w, h, objScore, c1, c2, ...] for grid[0,0]'s 3rd BBox
      [cx, cy, w, h, objScore, c1, c2, ...] for grid[0,1]'s 1st BBox
      [cx, cy, w, h, objScore, c1, c2, ...] for grid[0,1]'s 2nd BBox
      ...
    ]
    
    All in all, we don't really care which grid our detections are in, we just want the best ones
    Thus discarding this ease of access to grid cells is fine

"""


def flattenPredict(predictions, inpDim, anchors, numClasses, CUDA):
    # TODO: Add in parameter types

    """
    Flattens the feature map into 2D Tensor and returns predictions for all the bounding boxes/anchors

    :param predictions: the feature map being plugged into the yolo layer
    :param inpDim: size of feature map
    :param anchors: the anchors being used in the layer
    :param numClasses: number of classes to detect
    :param CUDA: If the computer has a CUDA enabled GPU
    :return: A tensor of reformatted predictions
    :rtype torch.Tensor
    """

    batchSize = predictions.size(0)
    stride = inpDim // predictions.size(2)
    gridSize = inpDim // stride
    bboxAttrs = 5 + numClasses
    numAnchors = len(anchors)

    # The dimension of the anchors is related to the height and width in the net block which describe image input size
    # These dimensions are larger by a factor of stride than the detection map, so we divide the anchors by the stride
    # This lets the anchors have the same relative dimension as the feature map we are working with
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    predictions = predictions.view(batchSize, bboxAttrs * numAnchors, gridSize * gridSize)
    predictions = predictions.transpose(1, 2).contiguous()
    predictions = predictions.view(batchSize, gridSize * gridSize * numAnchors, bboxAttrs)

    # ----------------------------------------------
    #  Perform the prediction step of the yolo layer
    # ----------------------------------------------

    # ---------------------------------------------------------------------
    #  Sigmoid activate the bbox x,y coords and the object confidence score
    # ---------------------------------------------------------------------

    # Remember we've reshaped the input feature map to be a 2D tensor
    predictions[:, :, 0] = torch.sigmoid(predictions[:, :, 0])  # Center x
    predictions[:, :, 1] = torch.sigmoid(predictions[:, :, 1])  # Center y
    predictions[:, :, 4] = torch.sigmoid(predictions[:, :, 4])  # Object confidence

    # ---------------------------------------------------------------------
    #  Add center offsets to the bbox x,y coords
    # ---------------------------------------------------------------------

    grid = np.arange(gridSize)
    gridx, gridy = np.meshgrid(grid, grid)

    xOffset = torch.FloatTensor(gridx).view(-1, 1)
    yOffset = torch.FloatTensor(gridy).view(-1, 1)

    if CUDA:
        xOffset = xOffset.cuda()
        yOffset = yOffset.cuda()

    xyOffset = torch.cat((xOffset, yOffset), 1).repeat(1, numAnchors).view(-1, 2).unsqueeze(0)

    predictions[:, :, :2] += xyOffset

    # ---------------------------------------------------------------------
    #  Scale bounding box with anchors
    # ---------------------------------------------------------------------

    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(gridSize * gridSize, 1).unsqueeze(0)
    predictions[:, :, 2:4] = torch.exp(predictions[:, :, 2:4]) * anchors

    # ---------------------------------------------------------------------
    #  Sigmoid class scores to get final class prediction
    # ---------------------------------------------------------------------

    # FIXME: Zi's NOTES - Softmax vs Sigmoid
    #  The source says to softmax this but they have a sigmoid activation in their source
    #  I'm going to stick with sigmoid, but if weird bugs come up this may be a line of contention
    predictions[:, :, 5:5 + numClasses] = torch.sigmoid((predictions[:, :, 5:5 + numClasses]))

    # ---------------------------------------------------------------------
    #  Resize feature map to size of input image.
    # ---------------------------------------------------------------------

    # This will make it so we can reference the bounding box attributes in relation to the size of the input image
    predictions[:, :, :4] *= stride

    return predictions


def predict(predictions, inpDim, anchors, numClasses, CUDA):
    # TODO Add in parameter types as well as return related docs
    """
    This function is nearly equivalent to flattenPredict but for the output of the final tensor.
    Whereas flattenPredict removes the grid index information, this keeps it.
    This is solely used during training steps

    :param predictions: the feature map being plugged into the yolo layer
    :param inpDim: size of feature map
    :param anchors: the anchors being used in the layer
    :param numClasses: number of classes to detect
    :param CUDA: If the computer has a CUDA enabled GPU

    """
    FloatTensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

    batchSize = predictions.size(0)
    stride = inpDim // predictions.size(2)
    gS = inpDim // stride
    nAttrs = 5 + numClasses
    nA = len(anchors)

    predictions = predictions.view(batchSize, nA, nAttrs, gS, gS).permute(0, 1, 3, 4, 2).contiguous()

    # Get outputs
    x = torch.sigmoid(predictions[..., 0])  # Center x
    y = torch.sigmoid(predictions[..., 1])  # Center y
    w = predictions[..., 2]  # Width
    h = predictions[..., 3]  # Height
    predConf = torch.sigmoid(predictions[..., 4])  # Conf
    predClass = torch.sigmoid(predictions[..., 5:])  # Cls pred.

    # Calculate offsets for each grid
    grid_x = torch.arange(gS).repeat(gS, 1).view([1, 1, gS, gS]).type(FloatTensor)
    grid_y = torch.arange(gS).repeat(gS, 1).t().view([1, 1, gS, gS]).type(FloatTensor)
    scaledAnchors = FloatTensor([(anchW / stride, anchH / stride) for anchW, anchH in anchors])
    anchor_w = scaledAnchors[:, 0:1].view((1, nA, 1, 1))
    anchor_h = scaledAnchors[:, 1:2].view((1, nA, 1, 1))

    # Add offset and scale with anchors
    predBoxes = FloatTensor(predictions[..., :4].shape)
    predBoxes[..., 0] = x.data + grid_x
    predBoxes[..., 1] = y.data + grid_y
    predBoxes[..., 2] = torch.exp(w.data) * anchor_w
    predBoxes[..., 3] = torch.exp(h.data) * anchor_h

    # FIXME: Put the following into the docs at some point
    # predBoxes = bounding box attrs of predictions
    # predConf = object confidence of predictions
    # predClass = class confidences of predictions
    return x, y, w, h, predBoxes, predConf, predClass, scaledAnchors


def findTrueDet(predictions, conf, numClasses, nmsThresh=0.4):
    """
    Performs object confidence thresholding and non-max suppression
    to find true object detections

    :param predictions: Tensor of all bounding box attributes
    :type predictions: torch.Tensor
    :param conf: objectness score threshold
    :param numClasses: number of classes
    :param nmsThresh: NMS IoU threshold
    :return: A tensor of all the bounding boxes' attributes which are true detections
    :rtype torch.Tensor
    """

    # -------------------------------
    #  Object confidence thresholding
    # -------------------------------
    confMask = (predictions[:, :, 4] > conf).float().unsqueeze(2)  # Mask BBoxes without sufficient obj conf score
    predictions = predictions * confMask  # Set the entire attribute row to 0

    # Check if there are any detections that haven't been zeroed
    # If there are no detections, we shouldn't go through rest of processing, so return 0 here
    try:
        ind_nz = torch.nonzero(predictions[:, :, 4]).transpose(0, 1).contiguous()
    except:
        return 0

    # ------------------------
    #  Non-maximum suppression
    # ------------------------

    # Reformat prediction array to contain the diagonal corners
    boxCorner = predictions.new(predictions.shape)
    boxCorner[:, :, 0] = (predictions[:, :, 0] - predictions[:, :, 2] / 2)  # top left x
    boxCorner[:, :, 1] = (predictions[:, :, 1] - predictions[:, :, 3] / 2)  # top left y
    boxCorner[:, :, 2] = (predictions[:, :, 0] + predictions[:, :, 2] / 2)  # bot right x
    boxCorner[:, :, 3] = (predictions[:, :, 1] + predictions[:, :, 3] / 2)  # bot right y
    predictions[:, :, :4] = boxCorner[:, :, :4]

    # NMS needs to be performed per image, so we iterate through the batch
    batchSize = predictions.size(0)
    output = predictions.new(1, predictions.size(2) + 1)
    haveDet = False  # Flag to check if we have any true detections

    for i in range(batchSize):
        imgPreds = predictions[i]  # Grab an individual image's predictions

        maxClass, maxClassInd = torch.max(imgPreds[:, 5:5 + numClasses], 1)  # We only care about class with highest score
        maxClass = maxClass.float().unsqueeze(1)  # We cast these two tensors as floats to reduce memory usage
        maxClassInd = maxClassInd.float().unsqueeze(1)
        nmsAttrs = (imgPreds[:, :5], maxClass, maxClassInd)  # We now only have 7 attributes per bounding box row to keep track of
        imgPreds = torch.cat(nmsAttrs, 1)

        # Get rid of bounding box rows which we had set to 0 earlier on
        nonZeroInd = (torch.nonzero(imgPreds[:, 4]))
        cleanedPreds = imgPreds[nonZeroInd.squeeze(), :].view(-1, 7)

        # FIXME: Zi's NOTES - Detections try catch
        #  I'm pretty sure this try except is useless since we catch a case for no detections at all above
        #  Since getting past that implies there are detections, we should always have at least one class
        try:
            # # Now we want to get all the classes found in the image since we do NMS class-wise
            # # There can be multiple true detections of same class
            # # We only need one, so we find all the unique classes
            #
            # tensor_np = cleanedPreds.cpu().numpy()
            # uniqueDets_np = np.unique(tensor_np)
            # uniqueDets_tens = torch.from_numpy(uniqueDets_np)
            #
            # # These lines are important for casting the tensor into either CPU or GPU backend
            # imgClasses = cleanedPreds.new(uniqueDets_tens.shape)
            # imgClasses.copy_(cleanedPreds)

            imgClasses = getUniques(cleanedPreds[:, -1])

        except:
            continue

        # Here is where we perform NMS
        for cls in imgClasses:
            # Get detections of one particular class
            clsMask = cleanedPreds * (cleanedPreds[:, -1] == cls).float().unsqueeze(1)
            clsMaskInd = torch.nonzero(clsMask[:, -2]).squeeze()
            classDets = cleanedPreds[clsMaskInd].view(-1, 7)

            # Sort with maximum objectness score at top
            sortMask = torch.sort(classDets[:, 4], descending=True)[1]
            classDets = classDets[sortMask]

            numDets = classDets.size(0)
            for j in range(numDets):

                # Get IOUs of all boxes after the one we are looking at
                try:
                    ious = getIOU(classDets[j].unsqueeze(0), classDets[j + 1:])
                except ValueError:
                    break
                except IndexError:
                    break

                # Zero out all detections with IOU < threshold
                iouMask = (ious > nmsThresh).float().unsqueeze(1)
                classDets[j + 1:] *= iouMask

                # Remove non-zero entries
                nonZeroInd = torch.nonzero(classDets[:, 4].squeeze())
                classDets = classDets[nonZeroInd].view(-1, 7)

            # classDets now holds all of the true detections in the image
            # We concat the batch id of the image into new column to help debugging
            batchInd = classDets.new(classDets.size(0), 1).fill_(i)
            temp = batchInd, classDets

            if not haveDet:
                output = torch.cat(temp, 1)
                haveDet = True
            else:
                out = torch.cat(temp, 1)
                output = torch.cat((output, out))

    return output


def getUniques(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def getIOU(mainBox, secBoxes, x1y1x2y2=True):
    """
    Returns the IoU of many bounding boxes in relation to one \n
    NOTE THE OPTIONAL PARAMETER:
    This is very important as it will tell the method if it will need to reformat the inputs
    By default the method was built to accept format in [x1, y1, x2, y2], so setting the parameter to False will cause it to reformat the inputs thus

    :param mainBox: Bounding box to calculate IOU from
    :type mainBox: torch.Tensor
    :param secBoxes: Secondary boxes. Tensor of bounding boxes whose IOU with the main box will be calculated
    :type secBoxes: torch.Tensor
    :param x1y1x2y2: Boolean to determine if function needs to reformat inputs
    :type x1y1x2y2: bool
    :return The IOU values of the secondary boxes in relation to the main box
    :rtype torch.Tensor
    """
    # Get the coordinates of bounding boxes

    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = mainBox[:, 0], mainBox[:, 1], mainBox[:, 2], mainBox[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = secBoxes[:, 0], secBoxes[:, 1], secBoxes[:, 2], secBoxes[:, 3]
    else:
        b1_x1, b1_x2 = mainBox[:, 0] - mainBox[:, 2] / 2, mainBox[:, 0] + mainBox[:, 2] / 2
        b1_y1, b1_y2 = mainBox[:, 1] - mainBox[:, 3] / 2, mainBox[:, 1] + mainBox[:, 3] / 2
        b2_x1, b2_x2 = secBoxes[:, 0] - secBoxes[:, 2] / 2, secBoxes[:, 0] + secBoxes[:, 2] / 2
        b2_y1, b2_y2 = secBoxes[:, 1] - secBoxes[:, 3] / 2, secBoxes[:, 1] + secBoxes[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    # FIXME: Zi's NOTES - I think 1e-16 is fix for floating point errors
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou
