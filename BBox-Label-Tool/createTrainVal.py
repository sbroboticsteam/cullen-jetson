import os
import shutil
import numpy as np
import itertools


# Create training and validation folders if there are none
# Then create image and label folders inside those
def createDirs():
    if os.path.exists("Train") and os.path.exists("Val"):
        return True
    else:
        os.makedirs("Train")
        os.makedirs(os.path.join("Train", "Images"))
        os.makedirs(os.path.join("Train", "Labels"))

        os.makedirs("Val")
        os.makedirs(os.path.join("Val", "Images"))
        os.makedirs(os.path.join("Val", "Labels"))

    # Create a negative folder for images which don't have any label associated with them
    if not os.path.exists("Neg"):
        os.makedirs("Neg")

    return False


updateTxts = createDirs()
classSplits = {}
imgs = np.array([])

# If both Training and Validation folders exist, we assume everything is already sorted
# Thus we'll just update the training and validation path text files
if updateTxts:
    print("Updating path files")
    with open("trainPath.txt", "w") as imgsFile:
        imgsRoot = os.path.join("Train", "Images")
        imgs = sorted(os.listdir(imgsRoot))

        for img in imgs:
            line = os.path.join(os.getcwd(), imgsRoot, img) + "\n"
            print(line)
            imgsFile.write(line)

    with open("valPath.txt", "w") as imgsFile:
        imgsRoot = os.path.join("Val", "Images")
        imgs = sorted(os.listdir(imgsRoot))

        for img in imgs:
            line = os.path.join(os.getcwd(), imgsRoot, img) + "\n"
            print(line)
            imgsFile.write(line)

else:
    # Get total images from every class and get the image names to prepare them to be split
    for folder in os.listdir("Images"):
        subPath = os.path.join("Images", folder)

        imgs = np.asarray([file for file in sorted(os.listdir(subPath))])
        numImgs = len(imgs)
        print(imgs)

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

                try:
                    print("{} --> {}".format(imgOrig, imgMove))
                    shutil.copy(imgOrig, imgMove)
                    print("{} --> {}".format(txtOrig, txtMove))
                    shutil.copy(txtOrig, txtMove)
                except FileNotFoundError:
                    print("{} Label not found. Moving to Negatives".format(imgMove))

        elif d[0] == "Val":
            imgsToMove = classSplits[d[1]][1]
            for i in imgsToMove:
                imgOrig = os.path.join("Images", d[1], i)
                imgMove = os.path.join(d[0], "Images", i)
                txtOrig = os.path.join("LabelsConverted", d[1], i.replace("jpg", "txt"))
                txtMove = os.path.join(d[0], "Labels", i.replace("jpg", "txt"))

                try:
                    print("{} --> {}".format(imgOrig, imgMove))
                    shutil.copy(imgOrig, imgMove)
                    print("{} --> {}".format(txtOrig, txtMove))
                    shutil.copy(txtOrig, txtMove)
                except FileNotFoundError:
                    print("{} Label not found. Moving to Negatives".format(imgMove))

    # Create txt with paths to training and validation images
    # We will assume the labels are the same name as images but under the Labels folder
    with open("trainPath.txt", "w") as imgsFile:
        imgsRoot = os.path.join("Train", "Images")
        imgs = sorted(os.listdir(imgsRoot))

        for img in imgs:
            line = os.path.join(os.getcwd(), imgsRoot, img) + "\n"
            print(line)
            imgsFile.write(line)

    with open("valPath.txt", "w") as imgsFile:
        imgsRoot = os.path.join("Val", "Images")
        imgs = sorted(os.listdir(imgsRoot))

        for img in imgs:
            line = os.path.join(os.getcwd(), imgsRoot, img) + "\n"
            print(line)
            imgsFile.write(line)

    # Move pictures with no annotations to negatives folder. Will not be used in training or validation
    for folder in os.listdir("LabelsConverted"):
        if not os.path.exists(os.path.join("Neg", folder)):
            os.makedirs(os.path.join("Neg", folder))

        subPath = os.path.join("Images", folder)
        imgs = sorted(os.listdir(subPath))
        for img in imgs:
            labelName = img.replace("jpg", "txt")
            labelFull = os.path.join("LabelsConverted", folder, labelName)

            if not os.path.exists(labelFull):
                # FIXME: If the number of images every go above 3 places, change this indexing
                imgNum = int(img[:3])
                fileFull = os.path.join("Images", folder, "{0:0=3d}.jpg".format(imgNum))

                if os.path.exists(fileFull):
                    imgMove = os.path.join("Neg", folder, "{0:0=3d}.jpg".format(imgNum))
                    shutil.copy(fileFull, imgMove)
