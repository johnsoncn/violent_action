"C:/Tools/miniconda3/envs/mmaction2/python.exe" "C:/Users/nmc_costa/Desktop/mmaction2/demo/webcam_demo.py" ^
C:/Users/nmc_costa/Desktop/mmaction2/configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py ^
https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth ^
C:/Users/nmc_costa/Desktop/mmaction2/demo/label_map_k400.txt ^
--average-size 5 ^
--threshold 0.2 


::Note: Considering the efficiency difference for users' hardware, Some modifications might be done to suit the case. Users can change:

::1). SampleFrames step (especially the number of clip_len and num_clips) of test_pipeline in the config file. 2). Change to the suitable Crop methods like TenCrop, ThreeCrop, CenterCrop, etc. in test_pipeline of the config file. 3). Change the number of --average-size. The smaller, the faster.



