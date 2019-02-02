import os
import cv2

origLblsPath = os.path.join("Labels", "001")
imgPath = os.path.join("Images", "001")
convPath = os.path.join("LabelsConverted", "001")

for txt in os.listdir(origLblsPath):
    filePath = os.path.join(origLblsPath, txt)
    outPath = os.path.join(convPath, txt)
    img = os.path.join(imgPath, txt.replace("txt", "jpg"))

    print(outPath)

    if os.path.exists(img):
        cvImg = cv2.imread(img)
        h, w, _ = cvImg.shape

        with open(filePath, "r") as f:
            content = f.readlines()
            content = [x.strip() for x in content]

            with open(outPath, "w+") as out:
                for i in content[1:]:
                    coords = i.split(" ")
                    coords = [int(x) for x in coords]
                    cx = ((coords[0] + coords[2]) / 2) / w
                    cy = ((coords[1] + coords[3]) / 2) / h
                    wNormed = abs(coords[0] - coords[2]) / w
                    hNormed = abs(coords[1] - coords[3]) / h

                    whCoords = [cx, cy, wNormed, hNormed]

                    out.write("0" + " " + " ".join([str(a) for a in whCoords]) + "\n")
