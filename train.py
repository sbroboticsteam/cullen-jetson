from __future__ import division

import os
from pytorch_yolo_v3.util.utils import *
from pytorch_yolo_v3.util.parse_config import *
from pytorch_yolo_v3.util.datasets import *
import darknet as dn

import torch
from torch.utils.data import DataLoader

# parser = argparse.ArgumentParser()
# parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
# parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
# parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
# parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
# parser.add_argument("--data_config_path", type=str, default="config/coco.data", help="path to data config file")
# parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
# parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
# parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
# parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
# parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
# parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
# parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
# parser.add_argument(
#     "--checkpoint_dir", type=str, default="checkpoints", help="directory where model checkpoints are saved"
# )
# parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
# opt = parser.parse_args()
# print(opt)

cuda = torch.cuda.is_available()

os.makedirs("output", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

classes = load_classes("pytorch_yolo_v3/data/tennisball.names")

paths = {
    "train": "Annotations/BBox-Label-Tool/trainPath.txt",
    "val": "Annotations/BBOX-Label-Tool/valPath.txt",
    "cfg": "pytorch_yolo_v3/cfg/yolov3-SBRT.cfg",
    "weights" : "pytorch_yolo_v3/weights/yolov3.weights",
    "checkpointDir" : "checkpoints"
}

# Get hyper parameters
hyperparams = parse_model_config(paths["cfg"])[0]
learningRate = float(hyperparams["learning_rate"])
momentum = float(hyperparams["momentum"])
decay = float(hyperparams["decay"])
burnIn = int(hyperparams["burn_in"])
batchSize = 4
nCPU = 0
epochs = 30
checkpointInterval = 1

# Initiate model
model = dn.Darknet(paths["cfg"])
# model.load_weights(paths["weights"])

if cuda:
    model = model.cuda()

model.train()

dataloader = torch.utils.data.DataLoader(
    ListDataset(paths["train"]), batch_size=batchSize, shuffle=False, num_workers=nCPU
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

for epoch in range(epochs):
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)

        optimizer.zero_grad()

        loss = model(imgs, targets)

        loss.backward()
        optimizer.step()

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
        model.save_weights("%s/%d.weights" % (paths["checkpointDir"], epoch))
