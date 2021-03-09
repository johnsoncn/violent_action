:: Testing for improved mAP(mean average precision) and Recall (Recall measures how well you find all the positives.).
::- inference with Time Augmentation (TTA) tutorial
::- inference with Model Ensembling Tutorial 


:: INFERENCE NORMALLY  
:: default settings testing 
"C:\\Tools\\miniconda3\\envs\\violent_action\\python.exe" "C:\\Users\\nmc_costa\\google_drive\\projects\\bosch_P19\\research\\python_ws\\violent_action\\yolo\\yolov5\\detect.py" ^
 --weights="yolov5s.pt" ^
 --img="640" ^
 --source="data\\images\\" ^
 --name="cocotesting\\detect\\normal" ^
 --project="D:\\external_datasets\\yolov5tests"


:: INFERENCE WITH TTA 
:: detect.py TTA inference operates identically to test.py TTA: simply append --augment and increase image size by 30% for improved mAP and recall to any existing detect.py command:
"C:\\Tools\\miniconda3\\envs\\violent_action\\python.exe" "C:\\Users\\nmc_costa\\google_drive\\projects\\bosch_P19\\research\\python_ws\\violent_action\\yolo\\yolov5\\detect.py" ^
 --weights="yolov5s.pt" ^
 --img="832" ^
 --source="data\\images\\" ^
 --augment ^
 --name="cocotesting\\detect\\tta" ^
 --project="D:\\external_datasets\\yolov5tests"

:: INFERENCE WITH ENSEMBLE
:: Append extra models to the --weights argument to run ensemble inference:
"C:\\Tools\\miniconda3\\envs\\violent_action\\python.exe" "C:\\Users\\nmc_costa\\google_drive\\projects\\bosch_P19\\research\\python_ws\\violent_action\\yolo\\yolov5\\detect.py" ^
 --weights="yolov5s.pt yolov5x.pt" ^
 --img="640" ^
 --source="data\\images\\" ^ 
 --name="cocotesting\\detect\\ensemble" ^
 --project="D:\\external_datasets\\yolov5tests"
 
 
pause

