# SBRT Vision Branch

## Network Information
Network being implemented is a PyTorch variant of Yolov3

Class being trained on in question are tennis balls

**IMPORTANT NOTES:** 

**This network was trained on BGR IMAGES.**

**This network EXPECTS NORMALIZED LABELS as input**

## Regarding Training
If you want to train the network on your own computer, first go into the Annotations folder
and delete the Pos and Val folders entirely. 

Then run the following:

```
python createTrainVal.py
```

This should recreate the trainPath.txt and valPath.txt which contain the absolute file paths to the training and validation
sets respectively in the aforementioned Pos and Val folders.

If you want to train on custom objects, use the BBox-Label-Tool to create all of your annotations. Then run the following:

```
python convert.py
python createTrainVal.py
```
