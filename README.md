# Object Detection Model Derckerisation

# Introduction 

In this repository we will deploy a pretrained model in a lightweight Docker container.


# Installation

Please follow these steps for proper installation with the appropriate modifications:

* Clone this repository: `git clone https://github.com/gost-sniper/x`
* Enter the local repo folder : `cd x`
* Build the Docker image : `docker build --build-arg APP_PORT=8080 -t <image-name> .`
* Run the Docker container : `docker run -p 8080:8080 -d <image-name>`
 
Now we should have our model deployed at `0.0.0.0:8080` (or `localhost:8080` for windows users)

# Demo

## Single image

While the model is deployed you can use the `demo-single.py` to test the model using one image:

```
usage: demo-single.py [-h] -i IMAGE

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGE, --image IMAGE
                        path to the input image
```

CLI: `python3 demo/demo-single.py  --image "path/to/image"`

## Multiple images 

While the model is deployed you can use the `demo-batch.py` to test the model using a folder filed with images:

```
usage: demo-batch.py [-h] -i INPUT [-o OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        path to the input folder
  -o OUTPUT, --output OUTPUT
                        path to the output folder
```

CLI: `python3 demo/demo-batch.py --input images --output output`