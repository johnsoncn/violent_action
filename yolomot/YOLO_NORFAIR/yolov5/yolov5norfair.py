import argparse

#import cv2 #METHOD 1
import numpy as np
import torch
import os
#from tool.torch_utils import do_detect #METHOD 1
#from tool.utils import load_class_names, plot_boxes_cv2 #METHOD 1

import norfair
from norfair import Detection, Tracker, Video



class YOLO:
    #YOLOv5 NC UPDATE
            #https://github.com/ultralytics/yolov5/issues/36
            #https: // pytorch.org / hub / ultralytics_yolov5 /
    def __init__(self, weightfile, use_cuda=True):
        if use_cuda and not torch.cuda.is_available():
            raise Exception(
                "Selected use_cuda=True, but cuda is not available to Pytorch"
            )
        self.use_cuda = use_cuda
        #SET MODEL
        #https://github.com/ultralytics/yolov5/issues/36
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=weightfile) # custom model load , default inference
        #To load a YOLOv5 model for training rather than inference, set autoshape=False. To load a model with randomly initialized weights (to train from scratch) use pretrained=False.
        if self.use_cuda:
            self.model.cuda()

    def __call__(self, img, conf=0.4, iou=0.6):
        """
        #METHOD 1 - resizing image, normalize and do_detect from tool.torch_utils import do_detect
        #NOT working - output = self.model(img) is fifferent and is difficult to extract the boxes
        #NOTE: Possibly the optimized way
        """
        # RESIZE
        # width, height = 416, 416
        # img = cv2.resize(img, (width, height))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # BOX DETECTION
        #boxes = do_detect(self.model, img, conf, iou, self.use_cuda)
        #return boxes[0]
        
        
        """
        #METHOD 2 - using pytorch.hub.model tools
        #working 
        #https://github.com/ultralytics/yolov5/issues/36
        """
        # NORMALIZE: 
        #img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0) #METHOD 2.1 TESTING normalization (Same output as METHOD 1)
        # CHANGE MODEL PARAMETERS:
        self.model.conf = conf
        self.model.iou = iou
        # INFERENCE:
        output = self.model(img)  # includes nms for each class; 
        #output = self.model(img, size=416) #resize is possible but the Boxes need to be converted back-how??
        # PRINT OUTPUT:
        #output.print()
        #output.show()  # or .save() or .show()
        #print(output.xyxy[0].cpu().detach().numpy())
        #      xmin    ymin    xmax   ymax  confidence  class
        #print(output[0].cpu().detach().numpy())
        # BOXES: 
        boxes=output.xyxy[0].cpu().detach().numpy() #NC the estimations seem not normalized
        #boxes=[[box[0]/255,box[1]/255,box[2]/255,box[3]/255,box[4],box[5]] for box in boxes] #METHOD 2.1 - revert
        print(boxes)
        return boxes #boxes[0]


def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)


def get_centroid(yolo_box, img_height, img_width):
    """METHOD 1"""
    # x1 = yolo_box[0] * img_width
    # y1 = yolo_box[1] * img_height
    # x2 = yolo_box[2] * img_width
    # y2 = yolo_box[3] * img_height
    """METHOD 2"""
    x1 = yolo_box[0] 
    y1 = yolo_box[1] 
    x2 = yolo_box[2] 
    y2 = yolo_box[3]
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Track objects in a video.")
    parser.add_argument("--weights", type=str, default="yolov5s.pt", help="wheights model path")
    parser.add_argument("--files", type=str, nargs="+", help="Video files to process")
    parser.add_argument("--conf", type=float, default="0.4",help="confidence threshold")
    parser.add_argument("--iou_thres", type=float, default="0.6",help="iou threshold")
    parser.add_argument("--debug",  action='store_true', default="0.6",help="iou threshold")
    args = parser.parse_args()
    
    model = YOLO(args.weights)  # set use_cuda=False if using CPU
    
    max_distance_between_points = 30
    for input_path in args.files:
        video = Video(input_path=input_path, output_path=os.path.dirname(input_path))
        tracker = Tracker(
            distance_function=euclidean_distance,
            distance_threshold=max_distance_between_points,
        )
        #tracker_c1 = Tracker(distance_function=euclidean_distance, distance_threshold=max_distance_between_points)
    
        for frame in video:
            detections = model(frame, conf=args.conf, iou=args.iou_thres) #__call__ method
            detections = [
                Detection(get_centroid(box, frame.shape[0], frame.shape[1]), data=box)
                for box in detections
                #if box[-1] == 2 #select the classes you want to track
            ]
            tracked_objects = tracker.update(detections=detections)
            norfair.draw_points(frame, detections)
            norfair.draw_tracked_objects(frame, tracked_objects)
            if (args.debug): norfair.draw_debug_metrics(frame, tracked_objects) #debug (optional)
            
            #detections_c1 = [Detection(get_centroid(box, frame.shape[0], frame.shape[1]), data=box) for box in detections if box[-1] == 1]
            #tracked_objects_c1 = tracker_c1.update(detections=detections_c1)
            #norfair.draw_points(frame, detections_c1, color="green")
            #norfair.draw_tracked_objects(frame, tracked_objects_c1, color="green")
            #norfair.draw_debug_metrics(frame, tracked_objects_c1) #debug
            
            video.write(frame)
            #video.show(frame)
