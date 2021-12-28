FROM python:3.7.9-slim

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt /
RUN pip3 install -r /requirements.txt

COPY ./app /app

WORKDIR /app

RUN python3 -c "from torchvision.models import detection; detection.retinanet_resnet50_fpn(pretrained=True, progress=True, pretrained_backbone=True)"

ARG APP_PORT
ENV PORT=$APP_PORT
EXPOSE ${PORT}
CMD gunicorn main:app -b 0.0.0.0:$PORT