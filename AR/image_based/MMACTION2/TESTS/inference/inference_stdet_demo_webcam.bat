"C:/Tools/miniconda3/envs/mmaction2/python.exe" "C:/Users/nmc_costa/Desktop/mmaction2/demo/webcam_demo_spatiotemporal_det.py" ^
--input-video 0 ^
--config C:/Users/nmc_costa/Desktop/mmaction2/configs/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb.py ^
--checkpoint https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth ^
--det-config C:/Users/nmc_costa/Desktop/mmaction2/demo/faster_rcnn_r50_fpn_2x_coco.py ^
--det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth ^
--det-score-thr 0.9 ^
--action-score-thr 0.5 ^
--label-map C:/Users/nmc_costa/Desktop/mmaction2/demo/label_map_ava.txt ^
--predict-stepsize 40 ^
--output-fps 20 ^
--show



