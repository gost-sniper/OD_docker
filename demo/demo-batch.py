import requests
import glob
import time
import numpy as np
import cv2
import pickle
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
	help="path to the input folder")
ap.add_argument("-o", "--output", type=str, default="output",
	help="path to the output folder")
args = vars(ap.parse_args())

COLORS = np.random.uniform(0, 255, size=(91, 3))
IDX = {v:k for k, v in pickle.loads(open("demo/coco_classes.pickle", "rb").read()).items()}

url = 'http://localhost:8080/api/detect'
images = glob.glob(f'{args["input"]}/*.jpg')

os.makedirs(args["output"], exist_ok=True)

for image in images:
    files = {'image': open(image,'rb')}

    since = time.time()
    r = requests.post(url, files=files)
    to = time.time() - since
    detections = r.json()["detections"]
    print(detections)
    orig = cv2.imread(image)
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

    cv2.imwrite(image.replace(args["input"], args['output']), orig)
    print(f"""{image.replace(f'{args["input"]}//', '')} processed in {to} secondes""")
        
