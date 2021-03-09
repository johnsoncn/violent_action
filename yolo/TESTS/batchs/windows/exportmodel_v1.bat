:: Export model
::pip install onnx>=1.7.0  # for ONNX export
::pip install coremltools==4.0  # for CoreML export
::info: https://github.com/ultralytics/yolov5/issues/251
"C:\\Tools\\miniconda3\\envs\\violent_action\\python.exe" "C:\\Users\\nmc_costa\\google_drive\\projects\\bosch_P19\\research\\python_ws\\violent_action\\yolo\\yolov5\\models\\export.py" ^
 --weights="D:\\external_datasets\\yolov5tests\\molatesting\\train\\exp\\weights\\best.pt" ^
 --img="640" ^
 --batch="8"


pause

