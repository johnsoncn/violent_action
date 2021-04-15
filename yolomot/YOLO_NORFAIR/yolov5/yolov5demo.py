import argparse

import cv2
import numpy as np
import torch
from tool.torch_utils import do_detect
from tool.utils import load_class_names, plot_boxes_cv2

import norfair
from norfair import Detection, Tracker, Video

max_distance_between_points = 30


class YOLO:
    def __init__(self, weightfile, use_cuda=True):
        if use_cuda and not torch.cuda.is_available():
            raise Exception(
                "Selected use_cuda=True, but cuda is not available to Pytorch"
            )
        self.use_cuda = use_cuda
        
        #SET MODEL
        #https://github.com/ultralytics/yolov5/issues/36
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=weightfile) # custom model load , default inference, pretrained=true
        #Number of Output classes: classes=80
        #To load a YOLOv5 model for training rather than inference, set autoshape=False. To load a model with randomly initialized weights (to train from scratch) use pretrained=False.
        if self.use_cuda:
            self.model.cuda()

    def __call__(self, img):
        #1.INIT image
        # width, height = 416, 416
        # img = cv2.resize(img, (width, height))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0) #This type of img input gives different output

        #https://github.com/ultralytics/yolov5/issues/36
        #https: // pytorch.org / hub / ultralytics_yolov5 /
        self.model.conf = 0.4
        self.model.iou = 0.6
        output = self.model(img)  # includes nms for each class
        #output.print()
        #output.show()  # or .save() or .show()
        #print(output.xyxy[0].cpu().detach().numpy())
        #      xmin    ymin    xmax   ymax  confidence  class
        #print(output[0].cpu().detach().numpy())

        boxes=output.xyxy[0].cpu().detach().numpy()
        #boxes=[[box[0]/255,box[1]/255,box[2]/255,box[3]/255,box[4],box[5]] for box in boxes]
        #boxes = do_detect(self.model, sized, 0.4, 0.6, self.use_cuda)
        print(boxes)
        return boxes #boxes[0]


def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)


def get_centroid(yolo_box, img_height, img_width):
    x1 = yolo_box[0] #* img_width
    y1 = yolo_box[1] #* img_height
    x2 = yolo_box[2] #* img_width
    y2 = yolo_box[3] #* img_height
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


parser = argparse.ArgumentParser(description="Track human poses in a video.")
parser.add_argument("--weights", type=str, default="yolov5s.pt", help="wheights model path")
parser.add_argument("--files", type=str, nargs="+", help="Video files to process")
args = parser.parse_args()

model = YOLO(args.weights)  # set use_cuda=False if using CPU

for input_path in args.files:
    video = Video(input_path=input_path)
    tracker = Tracker(
        distance_function=euclidean_distance,
        distance_threshold=max_distance_between_points,
    )

    for frame in video:
        detections = model(frame) #__call__ method
        detections = [
            Detection(get_centroid(box, frame.shape[0], frame.shape[1]), data=box)
            for box in detections
            if box[-1] == 2 #only cars
        ]
        tracked_objects = tracker.update(detections=detections)
        norfair.draw_points(frame, detections)
        norfair.draw_tracked_objects(frame, tracked_objects)
        video.write(frame)
