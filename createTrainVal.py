import os
import shutil
import numpy as np
import itertools


# Create training and validation folders if there are none
# Then create image and label folders inside those

def createDirs():
    if not os.path.exists("Train"):
        os.makedirs("Train")
    if not os.path.exists(os.path.join("Train", "Images")):
        os.makedirs(os.path.join("Train", "Images"))
    if not os.path.exists(os.path.join("Train", "Labels")):
        os.makedirs(os.path.join("Train", "Labels"))

    if not os.path.exists("Val"):
        os.makedirs("Val")
    if not os.path.exists(os.path.join("Val", "Images")):
        os.makedirs(os.path.join("Val", "Images"))
    if not os.path.exists(os.path.join("Val", "Labels")):
        os.makedirs(os.path.join("Val", "Labels"))

    # Create a negative folder for images which don't have any label associated with them
    if not os.path.exists("Neg"):
        os.makedirs("Neg")


createDirs()
classSplits = {}
imgs = np.array([])
# Get total images from every class and get the image names to prepare them to be split
for folder in os.listdir("Images"):
    subPath = os.path.join("Images", folder)

    imgs = np.array([file for file in os.listdir(subPath)])
    numImgs = len(imgs)

    # Dataset is split 80/20 training/validation
    trainInd = np.arange(0, int(numImgs * 0.8))
    valInd = np.arange(int(numImgs * 0.8), numImgs)

    trainImgs = imgs[trainInd]
    valImgs = imgs[valInd]
    classSplits.update({folder: (trainImgs, valImgs)})

# Setup and populate subdirectories within the training and validation folders
# This is when all the images are split into their respective folders
classes = classSplits.keys()
subDirs = list(itertools.product(["Train", "Val"], classes))

for d in subDirs:

    if d[0] == "Train":
        imgsToMove = classSplits[d[1]][0]
        for i in imgsToMove:
            imgOrig = os.path.join("Images", d[1], i)
            imgMove = os.path.join(d[0], "Images", i)
            txtOrig = os.path.join("LabelsConverted", d[1], i.replace("jpg", "txt"))
            txtMove = os.path.join(d[0], "Labels", i.replace("jpg", "txt"))

            print("{} --> {}".format(imgOrig, imgMove))
            shutil.move(imgOrig, imgMove)
            print("{} --> {}".format(txtOrig, txtMove))
            shutil.move(txtOrig, txtMove)

    elif d[0] == "Val":
        imgsToMove = classSplits[d[1]][1]
        for i in imgsToMove:
            imgOrig = os.path.join("Images", d[1], i)
            imgMove = os.path.join(d[0], "Images", i)
            txtOrig = os.path.join("LabelsConverted", d[1], i.replace("jpg", "txt"))
            txtMove = os.path.join(d[0], "Labels", i.replace("jpg", "txt"))

            print("{} --> {}".format(imgOrig, imgMove))
            shutil.move(imgOrig, imgMove)
            print("{} --> {}".format(txtOrig, txtMove))
            shutil.move(txtOrig, txtMove)

# Create txt with paths to training and validation images
# We will assume the labels are the same name as images but under the Labels folder
with open("trainPath.txt", "w+") as imgsFile:
    imgsRoot = os.path.join("Train", "Images")
    for img in os.listdir(imgsRoot):
        imgsFile.write(os.path.join(os.getcwd(), imgsRoot, img) + "\n")

with open("valPath.txt", "w+") as imgsFile:
    imgsRoot = os.path.join("Val", "Images")
    for img in os.listdir(imgsRoot):
        imgsFile.write(os.path.join(os.getcwd(), imgsRoot, img) + "\n")

# Move pictures with no annotations to negatives folder. Will not be used in training or validation
i = 0
for folder in os.listdir("LabelsConverted"):
    if not os.path.exists(os.path.join("Neg", folder)):
        os.makedirs(os.path.join("Neg", folder))

    subPath = os.path.join("LabelsConverted", folder)
    for txt in os.listdir(subPath):

        if int(txt[:3]) != i:
            fileFull = os.path.join("Images", folder, "{0:0=3d}.jpg".format(i))

            if os.path.exists(fileFull):
                imgMove = os.path.join("Neg", folder, "{0:0=3d}.jpg".format(i))
                shutil.move(fileFull, imgMove)
            i += 1
        i += 1
