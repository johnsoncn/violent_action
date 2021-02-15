REM START json2yolo

@echo OFF
title json2yolo

echo Hello! This is a batch to start json2yolo.

REM ACTIVATE ENV ********
call C:\Tools\miniconda3\condabin\conda.bat activate violent_action

REM change dir to start json2yolo*****
cd "C:\Users\nmc_costa\google_drive\projects\bosch_P19\research\python_ws\violent_action\yolo\json2yolo"

REM start json2yolo*****
set source=mola
set datasets_root_dir=D:\external_datasets\
set json_dir=D:\external_datasets\MOLA\annotations\train_mola_mix_aggressive\
set outdir=D:\external_datasets\MOLA\yoloformat\mola128\
set img_number=128
set copy_images=1

call python json2yolo.py --source=mola --datasets_root_dir=%datasets_root_dir% --json_dir=%json_dir% --outdir=%outdir% --img_number=%img_number% --copy_images=%copy_images%



pause