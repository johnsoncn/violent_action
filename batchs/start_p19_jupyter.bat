REM START VIOLENT ACTION ENV

@echo OFF
title violent_action

echo Hello! This is a batch to start violent_action jupyter.


REM ACTIVATE ENV ********
call C:\Tools\miniconda3\condabin\conda.bat activate violent_action

REM change dir to start pyff *****
cd "C:\Users\nmc_costa\google_drive\projects\bosch_P19\research\python_ws\violent_action"

REM Start jupyter lab *******
call jupyter lab

