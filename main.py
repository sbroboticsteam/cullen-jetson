from __future__ import division

import pickle as pkl
import random
import time

from darknet import Darknet
from utils.util import *
from zedstreamer import ZedCamera


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    # Return original dimension in (width, height) format. cv2's .shape is in (height, width) format
    dimWH = origImg.shape[1], origImg.shape[0]
    resized = cv2.resize(origImg, (inp_dim, inp_dim))
    resized = resized[:, :, ::-1].transpose((2, 0, 1)).copy()
    resized = torch.from_numpy(resized).float().div(255.0).unsqueeze(0)
    return resized, dimWH


def writeBBox(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);
    return img


if __name__ == '__main__':
    # Relative path to network weights and layer configuration
    cfgfile = "pytorch_yolo_v3/cfg/yolov3.cfg"
    weightsfile = "pytorch_yolo_v3/yolov3.weights"

    confidence = float(0.25)
    nms_thresh = float(0.4)
    hasCUDA = torch.cuda.is_available()

    num_classes = 80

    model = Darknet(cfgfile)
    model.load_weights(weightsfile)

    model.net_info["height"] = 160
    inp_dim = int(model.net_info["height"])

    # Darknet inputs need to be divisible by 32
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if hasCUDA:
        model.cuda()

    model.eval()

    # Start ZED camera
    zed = ZedCamera()

    frames = 0
    curTime = time.time()
    while True:

        origImg = zed.getImage("left")
        if origImg is not None:

            # ZED returns a 4D image. We just need the RGB section
            origImg = origImg[:, :, :3]

            # Resizes images to fit input dimension parameter. Also puts image into a torch tensor
            img, origDimWH = prep_image(origImg, inp_dim)

            if hasCUDA:
                img = img.cuda()

            output = model(Variable(img), hasCUDA)
            output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thresh)

            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.2f}".format(frames / (time.time() - curTime)))
                cv2.imshow("frame", origImg)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue

            output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(inp_dim)) / inp_dim

            # im_dim = im_dim.repeat(output.size(0), 1)
            output[:, [1, 3]] *= origDimWH[0]
            output[:, [2, 4]] *= origDimWH[1]

            classes = load_classes('pytorch_yolo_v3/data/coco.names')
            colors = pkl.load(open("pytorch_yolo_v3/pallete", "rb"))

            origImg = origImg.copy()
            list(map(lambda x: writeBBox(x, origImg), output))

            cv2.resize(origImg, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
            cv2.imshow("frame", origImg)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print("FPS of the video is {:5.2f}".format(frames / (time.time() - curTime)))


        else:
            break
