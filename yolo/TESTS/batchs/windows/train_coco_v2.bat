:: TRAIN DATA TEST [COCO]
:: This batch format not working - Tryed: pip install opencv_python ; pip install --upgrade --force-reinstall numpy

:: 1. TRAIN 
:: 1.1 Start tensorboard (optional)
::!rm -rfv "D:\\external_datasets\\yolov5tests\\cocotesting\\train" ::removing previous train
::load_ext tensorboard
::reload_ext tensorboard
::tensorboard --logdir "D:\\external_datasets\\yolov5tests\\cocotesting\\train"

:: 1.2 Weights & Biases Logging in realtime (optional)
:: During training you will see live updates at your account in [https://www.wandb.com/]

:: 1.3 Start training 
:: --weights '' =from scractch; --weights yolov5s.pt =from checkpoint
"C:\\Tools\\miniconda3\\envs\\violent_action\\python.exe" "C:\\Users\\nmc_costa\\google_drive\\projects\\bosch_P19\\research\\python_ws\\violent_action\\yolo\\yolov5\\train.py" ^
 --img="640" ^
 --batch="8" ^
 --epochs="3" ^
 --data="C:\\Users\\nmc_costa\\google_drive\\projects\\bosch_P19\\research\\python_ws\\violent_action\\yolo\\coco128_external.yaml" ^
 --cfg="" ^
 --weights="yolov5s.pt" ^
 --name="cocotesting\\train\\exp" ^
 --project="D:\\external_datasets\\yolov5tests"


pause

