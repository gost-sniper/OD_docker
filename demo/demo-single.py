# import the necessary packages
from torchvision.models import detection
import numpy as np
import argparse
import pickle
import torch
import cv2
import requests
import time

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to the input image")
args = vars(ap.parse_args())

# set the device we will be using to run the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load the list of categories in the COCO dataset and then generate a
# set of bounding box colors for each class
# CLASSES = pickle.loads(open(args["labels"], "rb").read())
np.random.seed(1)
COLORS = np.random.uniform(0, 255, size=(91, 3))
IDX = {v:k for k, v in pickle.loads(open("demo/coco_classes.pickle", "rb").read()).items()}


url = 'http://localhost:8080/api/detect'
files = {'image': open(args["image"], 'rb')}

since = time.time()
r = requests.post(url, files=files)
to = time.time() - since
detections = r.json()["detections"]

orig = cv2.imread(args["image"])

# loop over the detections
for i in range(0, len(detections["boxes"])):
    # extract the confidence (i.e., probability) associated with the
    # prediction
    confidence = detections["scores"][i]
    
    # extract the index of the class label from the detections,
    # then compute the (x, y)-coordinates of the bounding box
    # for the object
    idx = IDX.get(detections["labels"][i])
    box = detections["boxes"][i].values()
    (startX, startY, endX, endY) = list(box)
    # display the prediction to our terminal
    label = "{}: {:.2f}%".format(detections["labels"][i], confidence)
    print("[INFO] {}".format(label))
    # draw the bounding box and label on the image
    cv2.rectangle(orig, (startX, startY), (endX, endY),
        COLORS[idx], 2)
    y = startY - 15 if startY - 15 > 15 else startY + 15
    cv2.putText(orig, label, (startX, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
# show the output image
cv2.imshow("original", cv2.imread(args["image"]))
cv2.imshow("Output", orig)
cv2.waitKey(0)