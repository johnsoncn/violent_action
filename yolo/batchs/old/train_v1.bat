REM START Train yolov5

@echo OFF
title yolov5_train

echo Hello! This is a batch to start json2yolo.

REM ACTIVATE ENV ********
call C:\Tools\miniconda3\condabin\conda.bat activate violent_action

REM change dir to start yolov5 train*****
cd "C:\Users\nmc_costa\google_drive\projects\bosch_P19\research\python_ws\violent_action\yolo\yolov5"

REM start train*****
set img=640
set batch=8
set epochs=3
set data=C:\Users\nmc_costa\google_drive\projects\bosch_P19\research\python_ws\violent_action\yolo\mola128_external.yaml
set cfg=C:\Users\nmc_costa\google_drive\projects\bosch_P19\research\python_ws\violent_action\yolo\yolov5s.yaml
set weights=''
:: --weights '' =from scractch - necessary for new classes; --weights yolov5s.pt =from checkpoint

python train.py --img %img% --batch %batch% --epochs %epochs% --data %data% --cfg %cfg% --weights %weights%



pause