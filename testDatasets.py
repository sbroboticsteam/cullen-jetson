import matplotlib.pyplot as plt
import numpy as np
from pytorch_yolo_v3.util.datasets import ListDataset



def showBBox(img, bboxTensor):
    plt.imshow(img)

    for i in bboxTensor:
        if i.sum() != 0:
            print(i)





fig = plt.figure()
dataset = ListDataset("Annotations/BBox-Label-Tool/trainPath.txt")
print(len(dataset))


for i in range(10):
    imgPath, img, lblTensor = dataset[i+176]
    img = np.transpose(img)

    print(i, img.shape, lblTensor.shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    showBBox(img, lblTensor)

    if i == 3:
        plt.show()
        break