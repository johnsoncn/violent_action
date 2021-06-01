"/home/administrator/miniconda3/envs/mmaction2/bin/python" "/home/administrator/Z/Algorithms/mmaction2/demo/demo_spatiotemporal_det.py" \
--video /home/administrator/Z/Work/EASYRIDE/P19/NC/SAMPLES/MOLA_2P_BACK_MP4_SMALL/C3_P2_P1_2.mp4 \
--config /home/administrator/Z/Algorithms/mmaction2/configs/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb.py \
--checkpoint https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth \
--det-config /home/administrator/Z/Algorithms/mmaction2/demo/faster_rcnn_r50_fpn_2x_coco.py \
--det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
--det-score-thr 0.09 \
--action-score-thr 0.05 \
--label-map /home/administrator/Z/Work/EASYRIDE/P19/NC/mmaction2/CUSTOM/label_map_ava_violent.txt \
--predict-stepsize 4 \
--output-stepsize 2 \
--output-fps 8 \
--out-filename $PWD/C3_P2_P1_2_stdet.mp4

#https://github.com/open-mmlab/mmaction2/tree/master/demo#video-demo

#input video 


# Supported methods for Spatial Temporal Action Detection:

# Slowonly on Ava - Train Custom Classes From Ava Dataset: https://github.com/open-mmlab/mmaction2/blob/master/configs/detection/ava/README.md
# /home/administrator/Z/Algorithms/mmaction2/configs/recognition/slowonly/slowonly_r50_video_inference_4x16x1_256e_kinetics400_rgb.py
#https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r50_4x16x1_20e_ava_rgb/slowonly_omnisource_pretrained_r50_4x16x1_20e_ava_rgb_20201217-0c6d2e98.pth
#/home/administrator/Z/Algorithms/mmaction2/demo/label_map_ava.txt
#

#--label-map /home/administrator/Z/Algorithms/mmaction2/demo/label_map_ava.txt \
#--label-map /home/administrator/Z/Work/EASYRIDE/P19/NC/mmaction2/CUSTOM/label_map_ava_violent.txt \




