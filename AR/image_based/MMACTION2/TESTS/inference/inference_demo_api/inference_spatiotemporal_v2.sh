mkdir $PWD/"results"
for i in /home/administrator/Z/Work/EASYRIDE/P19/NC/SAMPLES/MOLA_2P_BACK_MP4_SMALL/*.mp4;
  do name="$(basename -s .mp4 $i)"
  echo "$name"
  "/home/administrator/miniconda3/envs/mmaction2/bin/python" "/home/administrator/Z/Algorithms/mmaction2/demo/demo_spatiotemporal_det.py" \
--video "$i" \
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
--out-filename $PWD/"results/${name}_sted.mp4" ;
done




