import os

import torch
import torch.multiprocessing as mp
from torch.autograd import Variable
from torch.utils.data import DataLoader

from darknet import Darknet
from utils.imgUtil import ListDataset
from utils.txtUtil import loadClasses
from utils.txtUtil import parse_data
import matplotlib.pyplot as plt

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
plt.ion()
plt.show()


def train(pid, model, data):

    model.loadWeights(data["weights"])
    # model.apply(initWeightsNormal)

    torch.manual_seed(int(data["seed"]) + pid)

    inpDim = int(data["reso"])
    epochs = int(data["epochs"])
    checkpointInterval = int(data["checkpoint_interval"])

    dataloader = DataLoader(
        ListDataset(data["train"], img_size=inpDim), batch_size=int(model.netInfo["batch"]), shuffle=False, num_workers=1
    )

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    for epoch in range(epochs):
        model.train()

        totalLoss = 0

        for batch_i, (_, imgs, labels) in enumerate(dataloader):
            imgs = Variable(imgs.type(Tensor))
            labels = Variable(labels.type(Tensor), requires_grad=False)

            optimizer.zero_grad()

            loss = model(imgs, labels)

            loss.backward()
            optimizer.step()

            totalLoss = loss.item()
            print("PROCESS_ID: %d --- [Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]" %
                  (
                      pid,
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

        if pid == int(data["num_processes"]) - 1:
            print(pid)
            plt.scatter(model.seen, totalLoss)
            plt.pause(0.05)
            if epoch % checkpointInterval == 0:
                print("Checkpointing")
                model.saveWeights("{}/Epoch_{}-P{}.weights".format(data["checkpoint_dir"], pid, epoch, round(totalLoss, 2)))


    return


if __name__ == '__main__':
    dat = parse_data("data/tennisball_TEST.data")

    if not os.path.exists(dat["checkpoint_dir"]):
        os.mkdir(dat["checkpoint_dir"])

    useCuda = dat["use_cuda"] and torch.cuda.is_available()
    device = torch.device("cuda" if useCuda else "cpu")

    featureExtract = bool(dat["feature_extract"])
    modelProcesses = int(dat["num_processes"])

    torch.manual_seed(dat["seed"])
    mp.set_start_method('spawn')

    model = Darknet(dat["cfg"], feature_extract=featureExtract)
    model.to(device)
    model.share_memory()

    # For some reason dat is passed as two arguments, so train has a buffer variable
    mp.spawn(fn=train, args=(model, dat), nprocs=modelProcesses)
    # processes = []
    # for rank in range(modelProcesses):
    #     # p = mp.Process(target=train, args=(model, data, rank))
    #     p = np.spawn(fn=train, args=(model, data, rank), )
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     print("joining")
    #     p.join()
