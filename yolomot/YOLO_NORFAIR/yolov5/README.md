# YOLOv5 example

Simplest possible example of tracking. Based on [pytorch YOLOv5](https://github.com/ultralytics/yolov5/issues/36).

## Instructions

1. Install Norfair with `pip install norfair[video]`.
2. (optional) Clone [yolov5](https://github.com/ultralytics/yolov5.git) and download the weights using wheighs/donwload_weights.sh (or use your onw pretrained model wheights)
3. Run `python yolov5demo.py files=<video file> wheights=<wheights model file> `.

## Explanation

This example tracks objects using a single point per detection: the centroid of the bounding boxes around cars returned by YOLOv5.

![Norfair YOLOv4 demo](../../docs/yolo_cars.gif)
