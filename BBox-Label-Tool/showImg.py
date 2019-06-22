import cv2
import numpy as np

if __name__ == '__main__':
    imgPath = "Train/Images/033.jpg"
    lblPath = "Train/Labels/033.txt"

    with open(lblPath, "r") as file:
        contents = file.readlines()

        attrs = []
        for line in contents:
            attrs.append([float(num) for num in line.split(" ")])

    img = cv2.imread(imgPath)
    h, w, _ = img.shape

    for bbox in attrs:
        cx = bbox[1]
        cy = bbox[2]
        wBoxNormed = bbox[3]
        hBoxNormed = bbox[4]

        cx = cx * w
        cy = cy * h
        wBox = wBoxNormed * w
        hBox = hBoxNormed * h

        x1 = int(cx - (wBox / 2))
        y1 = int(cy - (hBox / 2))
        x2 = int(cx + (wBox / 2))
        y2 = int(cy + (hBox / 2))

        print(x1, y1, x2, y2)

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(0)
