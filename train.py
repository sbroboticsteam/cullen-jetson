from __future__ import division

import os

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from darknet import Darknet
from utils.imgUtil import ListDataset
from utils.txtUtil import loadClasses
from utils.txtUtil import parse_data

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# ---------------------------------------------------
#           BEFORE YOU RUN ANYTHING!!!!!!
# ---------------------------------------------------
# Remember to change the height and width in the cfg file
# to be the same as the reso parameter in the data file

# If you are training with custom classes:
# Remember to change the classes param in the data file
#                    the classes param in the yolo layers of the cfg file
#                    the filters param in the layer before each yolo layer to be (5 + numClasses) * 3

# TODO: Change network to finetune
# Right now the network literally just retrains from scratch

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

    nCPU = 1

    model = Darknet(data["cfg"], featExtract=True)
    # model.apply(initWeightsNormal)
    model.loadWeights(data["weights"])

    if CUDA:
        model.cuda()

    model.train()

    dataloader = DataLoader(
        ListDataset(data["train"], img_size=inpDim), batch_size=int(model.netInfo["batch"]), shuffle=False, num_workers=nCPU
    )

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    for epoch in range(epochs):
        totalLoss = 0

        for batch_i, (_, imgs, labels) in enumerate(dataloader):
            imgs = Variable(imgs.type(Tensor))
            labels = Variable(labels.type(Tensor), requires_grad=False)

            optimizer.zero_grad()

            loss = model(imgs, labels)

            loss.backward()
            optimizer.step()

            totalLoss = loss.item()
            print("[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]" %
                  (
                      epoch,
                      epochs,
                      batch_i,
                      len(dataloader),
                      model.losses["x"],
                      model.losses["y"],
                      model.losses["w"],
                      model.losses["h"],
                      model.losses["conf"],
                      model.losses["cls"],
                      loss.item(),
                      model.losses["recall"],
                      model.losses["precision"],
                  )
                  )

            model.seen += imgs.size(0)

        if epoch % checkpointInterval == 0:
            model.saveWeights("{}/epoch_{}-{}.weights".format(data["checkpoint_dir"], epoch, totalLoss))
