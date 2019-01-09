# SBRT Vision Branch

Network being implemented is a PyTorch variant of Yolov3 <br>
Class being trained on in question are tennis balls <br>
<br>
If you want to train the network on your own computer, first go into the Annotations folder
and delete the Pos and Val folders entirely. Then run the following:

```
python createTrainVal.py
```

This should recreate the trainPath.txt and valPath.txt which contain the absolute file paths to the training and validation
sets respectively in the aforementioned Pos and Val folders
