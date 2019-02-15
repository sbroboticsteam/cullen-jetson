import os

import torch
import torch.multiprocessing as mp
from torch.autograd import Variable
from torch.utils.data import DataLoader

from darknet import Darknet
from utils.imgUtil import ListDataset
from utils.txtUtil import loadClasses
from utils.txtUtil import parse_data

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def train(rank):
    torch.manual_seed(data["seed"] + rank)

    dataloader = DataLoader(
        ListDataset(data["train"], img_size=inpDim), batch_size=int(model.netInfo["batch"]), shuffle=False, num_workers=1
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


if __name__ == '__main__':
    data = parse_data("data/tennisball_TEST.data")

    if not os.path.exists(data["checkpoint_dir"]):
        os.mkdir(data["checkpoint_dir"])

    confidence = float(data["confidence"])
    nmsThresh = float(data["nms_thresh"])

    epochs = int(data["epochs"])
    checkpointInterval = int(data["checkpoint_interval"])
    classes = loadClasses(data["names"])

    CUDA = torch.cuda.is_available() and bool(data["use_cuda"])
    device = torch.device("cuda" if CUDA else "cpu")
    inpDim = int(data["reso"])
    numClasses = data["classes"]
    featureExtract = data["feature_extract"]

    modelProcesses = data["num_processes"]
    torch.manual_seed(data["seed"])

    mp.set_start_method('spawn')

    model = Darknet(data["cfg"], feature_extract=featureExtract)
    model.to(device)
    model.share_memory()

    # model.apply(initWeightsNormal)
    model.loadWeights(data["weights"])

    if CUDA:
        model.cuda()

    model.train()

    processes = []

    for rank in range(modelProcesses):
        p = mp.Process(target=train, args=(rank))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
