# Yolov5 + Deep Sort with PyTorch




## Introduction

This repository contains a moded version of PyTorch YOLOv5 (https://github.com/ultralytics/yolov5). 

It filters out every detection that is not a person. The detections of persons are then passed to a Deep Sort algorithm (https://github.com/ZQPei/deep_sort_pytorch) which tracks the persons. The reason behind the fact that it just tracks persons is that the deep association metric is trained on a person ONLY datatset.

## Description

The implementation is based on two papers:

- Simple Online and Realtime Tracking with a Deep Association Metric
https://arxiv.org/abs/1703.07402
- YOLOv4: Optimal Speed and Accuracy of Object Detection
https://arxiv.org/pdf/2004.10934.pdf

## Requirements

Python 3.8 or later with all requirements.txt dependencies installed, including torch>=1.6. To install run:

`pip install -U -r requirements.txt`

All dependencies are included in the associated docker images. Docker requirements are: 
- `nvidia-docker`
- Nvidia Driver Version >= 440.44

## Before you run the tracker

1. Clone the repository recursively:

`git clone --recurse-submodules https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch.git`

If you already cloned and forgot to use `--recurse-submodules` you can run `git submodule update --init`

2. Github block pushes of files larger than 100 MB (https://help.github.com/en/github/managing-large-files/conditions-for-large-files). Hence you need to download two different weights: the ones for yolo and the ones for deep sort

- download the yolov5 weight from the latest realease https://github.com/ultralytics/yolov5/releases. Place the downlaoded `.pt` file under `yolov5/weights/`
- download the deep sort weights from https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6. Place ckpt.t7 file under`deep_sort/deep/checkpoint/`

## Tracking On Screen

The original tracking can be run on most video formats, but I created a new one which can be applied to many Industrial scenarios, that is analyze a live streaming video displayed on your screen.


```bash
tracking_on_screen(trajectory).py
```
#### Parameter Interpretation


--weights: \
Path of the model you want to deploy. You can also train your YOLOV5 model and apply it here. Default is 'yolov5/weights/yolov5s.pt'

--config-deepsort: \
A yaml file path that contains all the deepsort setup. Default is "deep_sort_pytorch/configs/deep_sort.yaml"

--Record: \
By default, the tracking and detecting results would not be saved into a text file as a format of continuous numpy arrays.

--record-path: \
Path for the result text file. Default is 'inference/output'

--record-name: \
Result text file name. Default is 'tracking_record.txt'

--device: \
Cuda device, gpu 0 or 1, 2, 3. Or if you are running without cuda, you should set it to cpu

--conf-thres: \
Object confidence threshold

--iou-thres: \
IOU threshold for NMS

--classes
Choose the corresponding class id that you want to track in you model. Default is 0.

The following 4 parameters defines the region in your computer monitor.\
--region-x1: Default is 0. \
--region-y1: Default is 175. \
--region-x2: Default is 928.\
--region-y2: Default is 687.

The following 2 parameters defines how you want to resize the images for processing. For the constriction of resize-width and resize-height, you should also need to check the 11-23-2020 report for instruction. \
--resize-width: Default is 928.\
--resize-height: Default is 512.



## Other information

For more detailed information about the algorithms and their corresponding lisences used in this project access their official github implementations.

