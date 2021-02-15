:: TRAIN CUSTOM DATA [MOLA]
:: This batch format not working - Tryed: pip install opencv_python ; pip install --upgrade --force-reinstall numpy; try from anaconda prompt


:: 1. CONFIGURE DATA TO YOLOV5 FORMAT 
:: 1.1 Create labels and organize dataset in YOLO FORMAT
::"C:\\Tools\\miniconda3\\envs\\violent_action\\python.exe" ::"C:\\Users\\nmc_costa\\google_drive\\projects\\bosch_P19\\research\\python_ws\\violent_action\\yolo\\json2yolo\\json2yolo.py" ^
:: --source="mola" ^
:: --datasets_root_dir="D:\\external_datasets\\" ^
:: --json_dir="D:\\external_datasets\\MOLA\\annotations\\split_mola_fix_equal\\mix_aggressive\\" ^
:: --outdir="D:\\external_datasets\\MOLA\\yoloformat\\mola128\\" ^
:: --img_number="128" ^
:: --copy_images="1"
 
:: 1.2 Create dataset .yaml file (copy from coco_external.yaml)

:: 1.3 Select a Model(yolov5s.yaml, yolov5x.yaml, ...) and update the number of classes to match the dataset


:: 2. TRAIN 
:: 2.1 Start tensorboard (optional)
::!rm -rfv "D:\\external_datasets\\yolov5tests\\molatesting\\train" ::removing previous train
::load_ext tensorboard
::reload_ext tensorboard
::tensorboard --logdir "D:\\external_datasets\\yolov5tests\\molatesting\\train"

:: 2.2 Weights & Biases Logging in realtime (optional)
:: During training you will see live updates at your account in [https://www.wandb.com/]

:: 2.3 Start training 
:: --weights '' =from scractch; --weights yolov5s.pt =from checkpoint
"C:\\Tools\\miniconda3\\envs\\violent_action\\python.exe" "C:\\Users\\nmc_costa\\google_drive\\projects\\bosch_P19\\research\\python_ws\\violent_action\\yolo\\yolov5\\train.py" ^
 --img="640" ^
 --batch="8" ^
 --epochs="3" ^
 --data="C:\\Users\\nmc_costa\\google_drive\\projects\\bosch_P19\\research\\python_ws\\violent_action\\yolo\\mola128_external.yaml" ^
 --cfg="C:\\Users\\nmc_costa\\google_drive\\projects\\bosch_P19\\research\\python_ws\\violent_action\\yolo\\mola128_yolov5s.yaml" ^
 --weights="''" ^
 --name="molatesting\\train\\exp" ^
 --project="D:\\external_datasets\\yolov5tests"


pause

