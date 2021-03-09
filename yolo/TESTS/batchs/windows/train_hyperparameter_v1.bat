:: HYPERPAREMETER EVOLUTION
:: Hyperparameter evolution is a method of Hyperparameter Optimization using a Genetic Algorithm (GA) for optimization.

:: 1. INITIALIZE HYPERPARAMETERS

:: # Hyperparameters for COCO training from scratch : yolov5/data/hyp.scratch.yaml
:: lr0: 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3) 
:: lrf: 0.2  # final OneCycleLR learning rate (lr0 * lrf) 
:: momentum: 0.937  # SGD momentum/Adam beta1 
:: weight_decay: 0.0005  # optimizer weight decay 5e-4 
:: warmup_epochs: 3.0  # warmup epochs (fractions ok) 
:: warmup_momentum: 0.8  # warmup initial momentum 
:: warmup_bias_lr: 0.1  # warmup initial bias lr 
:: giou: 0.05  # box loss gain 
:: cls: 0.5  # cls loss gain 
:: cls_pw: 1.0  # cls BCELoss positive_weight 
:: obj: 1.0  # obj loss gain (scale with pixels) 
:: obj_pw: 1.0  # obj BCELoss positive_weight 
:: iou_t: 0.20  # IoU training threshold 
:: anchor_t: 4.0  # anchor-multiple threshold 
:: # anchors: 0  # anchors per output grid (0 to ignore) 
:: fl_gamma: 0.0  # focal loss gamma (efficientDet default gamma=1.5) 
:: hsv_h: 0.015  # image HSV-Hue augmentation (fraction) 
:: hsv_s: 0.7  # image HSV-Saturation augmentation (fraction) 
:: hsv_v: 0.4  # image HSV-Value augmentation (fraction) 
:: degrees: 0.0  # image rotation (+/- deg) 
:: translate: 0.1  # image translation (+/- fraction) 
:: scale: 0.5  # image scale (+/- gain) 
:: shear: 0.0  # image shear (+/- deg) 
:: perspective: 0.0  # image perspective (+/- fraction), range 0-0.001 
:: flipud: 0.0  # image flip up-down (probability) 
:: fliplr: 0.5  # image flip left-right (probability) 
:: mosaic: 1.0  # image mosaic (probability) 
:: mixup: 0.0  # image mixup (probability) 

:: 2. DEFINE FITNESS
:: In yolov5/utils/metrics.py we have defined a default fitness function as a weighted combination of metrics: mAP@0.5 contributes 10% of the weight and mAP@0.5:0.95 contributes the remaining 90%. You may adjust these as you see fit or use the default fitness definition.

:: 3. EVOLVE

:: 3.1 Base scenario: COCO128 for 10 epochs using pretrained YOLOv5s
"C:\\Tools\\miniconda3\\envs\\violent_action\\python.exe" "C:\\Users\\nmc_costa\\google_drive\\projects\\bosch_P19\\research\\python_ws\\violent_action\\yolo\\yolov5\\train.py" ^
 --img="640" ^
 --batch="16" ^
 --epochs="10" ^
 --data="C:\\Users\\nmc_costa\\google_drive\\projects\\bosch_P19\\research\\python_ws\\violent_action\\yolo\\coco128_external.yaml" ^
 --cfg="" ^
 --weights="yolov5s.pt" ^
 --name="cocotesting\\train128\\base" ^
 --project="D:\\external_datasets\\yolov5tests" ^
 --hyp="data\\hyp.scratch.yaml" ^
 --cache
 
:: 3.2 Evolve hyperparameters: maximize the fitness function
::The default evolution settings will run the base scenario 300 times inyolov5/train.py. Results apper in evolve.txt and evolve/hyp_evolved.yaml
::the evolve.txt is used to mutate the metadata scale[0-1] in yolov5/train.py.

:: 3.3.1 Single-gpu
"C:\\Tools\\miniconda3\\envs\\violent_action\\python.exe" "C:\\Users\\nmc_costa\\google_drive\\projects\\bosch_P19\\research\\python_ws\\violent_action\\yolo\\yolov5\\train.py" ^
 --img="640" ^
 --batch="16" ^
 --epochs="10" ^
 --data="C:\\Users\\nmc_costa\\google_drive\\projects\\bosch_P19\\research\\python_ws\\violent_action\\yolo\\coco128_external.yaml" ^
 --cfg="" ^
 --weights="yolov5s.pt" ^
 --name="cocotesting\\train128\\singlegpu" ^
 --project="D:\\external_datasets\\yolov5tests" ^
 --cache ^
 --hyp="data\\hyp.scratch.yaml"
 --evolve
 
:: 3.3.2 Multi-gpu??
::# Multi-GPU
::for i in 0 1 2 3; do
::  nohup python train.py --epochs 10 --data coco128.yaml --weights yolov5s.pt --cache --evolve --device $i > evolve_gpu_$i.log &
::done

:: 4. VISUALIZE

:: Results are saved as yolov5/evolve.png, with one plot per hyperparameter. Values are on the x axis and fitness on the y axis. Yellow indicates higher concentrations. Vertical lines indicate that a parameter has been disabled and does not mutate. This is user selectable in the meta dictionary in train.py, and is useful for fixing parameters and preventing them from evolving.


pause

