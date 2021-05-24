"/home/administrator/miniconda3/envs/mmaction2/bin/python" "/home/administrator/Z/Algorithms/mmaction2/demo/demo_gradcam.py" \
/home/administrator/Z/Algorithms/mmaction2/configs/recognition/tsm/tsm_r50_video_inference_1x1x8_100e_kinetics400_rgb.py \
https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_video_1x1x8_100e_kinetics400_rgb/tsm_r50_video_1x1x8_100e_kinetics400_rgb_20200702-a77f4328.pth \
/home/administrator/Z/Work/EASYRIDE/P19/NC/SAMPLES/MOLA_2P_BACK_MP4_SMALL/C3_P2_P1_2.mp4 \
--target-layer-name backbone/layer4/1/relu \
--out-filename $PWD/C3_P2_P1_2_gradcam.gif

#https://github.com/open-mmlab/mmaction2/tree/master/demo#video-demo

#input video or rawframes




