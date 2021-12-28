#!/bin/python3
from flask import Flask, jsonify, make_response
from flask_restx import Api, Resource

from torchvision.models import detection
import numpy as np
import pickle
import torch
import cv2

from werkzeug.datastructures import FileStorage

app = Flask(__name__)
api = Api(app, doc='/docs', version='1.0', 
          title=' Wrapper for a Published Object-Detection Model',
          description='Object Detection API')


upload_parser = api.parser()
upload_parser.add_argument('image', location='files', 
                           type=FileStorage, required=True)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load the list of categories in the COCO dataset and then generate a
# set of bounding box colors for each class
CLASSES = pickle.loads(open("coco_classes.pickle", "rb").read())
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
CONFIDENCE = 0.5

model = detection.retinanet_resnet50_fpn(pretrained=True, progress=True,
	num_classes=len(CLASSES), pretrained_backbone=True).to(DEVICE)
model.eval()

@api.route('/api/detect', doc={
        "description": "Object Detection endpoint using pretrained model RetinaNet ",
    })
@api.expect(upload_parser)
class Predict(Resource):
    @api.response(200, 'Succes')
    @api.response(501, 'Invalid file format')
    def post(self):
        args = upload_parser.parse_args()
        uploaded_file = args['image'].read()
        #convert string data to numpy array
        npimg = np.fromstring(uploaded_file, np.uint8)
        # convert numpy array to image
        img = cv2.imdecode(npimg, cv2.cv2.IMREAD_COLOR)
        # print(type(uploaded_file))
        image = img
        try:
            orig = image.copy()
            # convert the image from BGR to RGB channel ordering and change the
            # image from channels last to channels first ordering
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.transpose((2, 0, 1))
            # add the batch dimension, scale the raw pixel intensities to the
            # range [0, 1], and convert the image to a floating point tensor
            image = np.expand_dims(image, axis=0)
            image = image / 255.0
            image = torch.FloatTensor(image)
            # send the input to the device and pass the it through the network to
            # get the detections and predictions
            image = image.to(DEVICE)
            detections = model(image)[0]
            # the return data structure placeholder
            result = {
                "detections": { 
                            "labels": [],
                            "scores": [], 
                            "boxes": [] 
                            }
                    }
            # loop over the detections
            for i in range(0, len(detections["boxes"])):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections["scores"][i]
                # filter out weak detections by ensuring the confidence is
                # greater than the minimum confidence
                if confidence > CONFIDENCE:
                    # extract the index of the class label from the detections,
                    # then compute the (x, y)-coordinates of the bounding box
                    # for the object
                    idx = int(detections["labels"][i])
                    box = detections["boxes"][i].detach().cpu().numpy()
                    (startX, startY, endX, endY) = box.astype("int")
                    result["detections"]["boxes"].append({"startX": int(startX), 
                                                          "startY": int(startY), 
                                                          "endX": int(endX), 
                                                          "endY": int(endY)})

                    result["detections"]["labels"].append(CLASSES[idx])
                    result["detections"]["scores"].append(round((confidence * 100).item(), 2))
            
            return make_response(jsonify(result), 200)
        except AttributeError:
            return "Please Provide an Image as input", 501


@api.route('/')
class Home(Resource):
    def get(self):
        return make_response(jsonify({"msg": "Please send your image to '/api/detect' to get your detection"}), 200)



if __name__ == '__main__':
    app.run(debug=True)