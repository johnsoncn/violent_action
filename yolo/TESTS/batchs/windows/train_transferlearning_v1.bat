:: TRANSFER LEARNING WITH FROZEN LAYERS
:: freeze YOLOv5 layers when transfer learning.

:: 1. FREEZE

:: 1.1 Freeze backbone
::All layers that match the freeze list in train.py will be frozen by setting their gradients to zero before training starts.
::check list of all model parameters: add to train.py before #freeze for k, v in model.named_parameters(): print(k) 
::check the backbone in the models/model.yaml file the layers that need to be frozen
::Example yolov5/models/yolov5s.yaml, [9 first layers] frozen in train.py:  
::# Freeze
::freeze = ['model.%s.' % x for x in range(10)]  # parameter names to freeze (full or partial)

:: 1.1 Freeze all layers
::To freeze the full model except for the final output convolution layers in Detect(), we set freeze list to contain all modules with 'model.0.' - 'model.23.' in their names, and then we start training:
::# Freeze
::freeze = ['model.%s.' % x for x in range(24)]  # parameter names to freeze (full or partial)

:: 2. RESULTS
::Train YOLOv5m on VOC on both of the above scenarios, along with a default model (no freezing, freeze = []), starting from the official COCO pretrained --weights yolov5m.pt. The training command for all runs was:
"C:\\Tools\\miniconda3\\envs\\violent_action\\python.exe" "C:\\Users\\nmc_costa\\google_drive\\projects\\bosch_P19\\research\\python_ws\\violent_action\\yolo\\yolov5\\train.py" ^
 --img="512" ^
 --batch="48" ^
 --epochs="50" ^
 --data="data\\voc.yaml" ^
 --cfg="" ^
 --weights="yolov5m.pt" ^
 --name="voctesting\\train\\tlearning" ^
 --project="D:\\external_datasets\\yolov5tests" ^
 --hyp="data\\hyp.finetune.yaml" ^
 --cache

::yolov5 report: https://wandb.ai/glenn-jocher/yolov5_tutorial_freeze/reports/Freezing-Layers-in-YOLOv5--VmlldzozMDk3NTg 
::P19 report:

pause