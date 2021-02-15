:: [MOLA] Testing for improved mAP(mean average precision) and Recall (Recall measures how well you find all the positives.).
::- Test Time Augmentation (TTA) tutorial
::- Test Model Ensembling Tutorial 
::- Test Pruning/Sparsity Tutorial.


:: TEST NORMALLY  
:: default settings testing 
"/home/administrator/C/Tools/miniconda3/envs/violent_action/python.exe" "/home/administrator/C/Users/nmc_costa/google_drive/projects/bosch_P19/research/python_ws/violent_action/yolo/yolov5/test.py" \
 --weights="/home/administrator/D/external_datasets/yolov5tests/molatesting/train/exp/weights/best.pt" \
 --img="640" \
 --data="/home/administrator/C/Users/nmc_costa/google_drive/projects/bosch_P19/research/python_ws/violent_action/yolo/coco_external.yaml" \
 --name="molatesting/test/normal" \
 --project="/home/administrator/D/external_datasets/yolov5tests"


:: TEST TTA 
:: append '--augment' and increase image size by 30% for improved mAP and recall. TTA enabled will typically take about 2-3X the time of normal inference as the images are being left-right flipped and processed at 3 different resolutions, with the outputs merged before NMS. Part of the speed decrease is simply due to larger image sizes (832 vs 640), while part is due to the actual TTA operations.
"/home/administrator/C/Tools/miniconda3/envs/violent_action/python.exe" "/home/administrator/C/Users/nmc_costa/google_drive/projects/bosch_P19/research/python_ws/violent_action/yolo/yolov5/test.py" \
 --weights="/home/administrator/D/external_datasets/yolov5tests/molatesting/train/exp/weights/best.pt" \
 --img="832" \
 --data="/home/administrator/C/Users/nmc_costa/google_drive/projects/bosch_P19/research/python_ws/violent_action/yolo/coco_external.yaml" \
 --augment \
 --name="molatesting/test/tta" \
 --project="/home/administrator/D/external_datasets/yolov5tests"

:: TEST MODEL ENSEMBLE
:: Multiple pretraind models may be ensembled together at test by simply appending extra models to the --weights. This example tests an ensemble of 2 models togethor: "yolov5s.pt yolov5x.pt".
"/home/administrator/C/Tools/miniconda3/envs/violent_action/python.exe" "/home/administrator/C/Users/nmc_costa/google_drive/projects/bosch_P19/research/python_ws/violent_action/yolo/yolov5/test.py" \
 --weights="/home/administrator/D/external_datasets/yolov5tests/molatesting/train/exp/weights/best.pt yolov5x.pt" \
 --img="640" \
 --data="/home/administrator/C/Users/nmc_costa/google_drive/projects/bosch_P19/research/python_ws/violent_action/yolo/coco_external.yaml" \
 --name="molatesting/test/ensemble" \
 --project="/home/administrator/D/external_datasets/yolov5tests"

:: TEST PRUNING/SPARSITY
:: We now repeat the above 'TEST NORMALLY' with a pruned model. Any model may be pruned after loading with the torch_utils.prune() command. In this example we prune a model after loading it for testing in test.py. We update test.py as: 1)go to yolov5.test.test() and 2) add before #configure the command torch_utils.prune(model. 0.3)
::"/home/administrator/C/Tools/miniconda3/envs/violent_action/python.exe" "/home/administrator/C/Users/nmc_costa/google_drive/projects/bosch_P19/research/python_ws/violent_action/yolo/yolov5/test.py" \
:: --weights="/home/administrator/D/external_datasets/yolov5tests/molatesting/train/exp/weights/best.pt" \
:: --img="640" \
:: --data="/home/administrator/C/Users/nmc_costa/google_drive/projects/bosch_P19/research/python_ws/violent_action/yolo/coco_external.yaml" \
:: --name="molatesting/test/pruning" \
:: --project="/home/administrator/D/external_datasets/yolov5tests"



pause

