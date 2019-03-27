from __future__ import division

import os

import torch
from torch.utils.data import DataLoader

from utils.imgUtil import ListDataset
from utils.txtUtil import loadClasses
from utils.txtUtil import parse_data

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

if __name__ == '__main__':
    data = parse_data("data/tennisball.data")

    if not os.path.exists(data["checkpoint_dir"]):
        os.mkdir(data["checkpoint_dir"])

    confidence = float(data["confidence"])
    nmsThresh = float(data["nms_thresh"])

    epochs = int(data["epochs"])
    checkpointInterval = int(data["checkpoint_interval"])
    classes = loadClasses(data["names"])

    CUDA = torch.cuda.is_available() and bool(data["use_cuda"])
    inpDim = int(data["reso"])
    numClasses = data["classes"]
    featExt = data["feature_extract"]

    nCPU = 1

    # model = Darknet(data["cfg"], feature_extract=featExt)
    # # model.apply(initWeightsNormal)
    # # model.loadWeight(data["weights"])
    # model.loadStateDict("weights/yolov3-320.pt")
    #
    # if CUDA:
    #     model.cuda()
    #
    # model.train()

    dataloader = DataLoader(
        ListDataset(data["train"], img_size=320), batch_size=4, shuffle=False, num_workers=nCPU
    )

    # plt.show()
    # plt.ion()
    # fig, axarr = plt.subplots(2, 2)
    # figB, axB = plt.subplots(2, 2)
    # figA, axA = plt.subplots(2, 2)

    for batch_i, (_, imgs, labels) in enumerate(dataloader):
        print()
        # seqDet = augments.to_deterministic()
        # imgs = np.transpose(imgs, (0, 2, 3, 1))
        # imgs = [np.flip(x, -1) for x in imgs]
        #
        # imgs = seqDet.augment_images([img for img in imgs])

        # axarr[0, 0].imshow(imgs[0])
        # axarr[0, 1].imshow(imgs[1])
        # axarr[1, 0].imshow(imgs[2])
        # axarr[1, 1].imshow(imgs[3])
        # plt.pause(0.05)
        # plt.waitforbuttonpress()
