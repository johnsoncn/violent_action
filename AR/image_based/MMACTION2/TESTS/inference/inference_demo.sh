"/home/administrator/miniconda3/envs/mmaction2/bin/python" "/home/administrator/Z/Algorithms/mmaction2/demo/demo.py" \
/home/administrator/Z/Algorithms/mmaction2/configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
"/home/administrator/Z/Work/EASYRIDE/P19/NC/SAMPLES/MOLA_2P_BACK/C3_P2_P1_2.avi" \
"/home/administrator/Z/Algorithms/mmaction2/demo/label_map_k400.txt" \
--font-size="30" \
--font-color="red" \
--out-filename $PWD/C3_P2_P1_2_demo.gif \
--target-resolution 340 -1

#https://github.com/open-mmlab/mmaction2/tree/master/demo#video-demo

#input video or rawframes


#Supported methods for Action Recognition:


#TSN on k400
#/home/administrator/Z/Algorithms/mmaction2/configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \´
#https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
#/home/administrator/Z/Algorithms/mmaction2/checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
#/home/administrator/Z/Algorithms/mmaction2/demo/label_map_k400.txt
#configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
#checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \

#R(2+1)D on K400 - best acc
#configs/recognition/r2plus1d/r2plus1d_r34_video_inference_8x8x1_180e_kinetics400_rgb.py \
#https://download.openmmlab.com/mmaction/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb/r2plus1d_r34_8x8x1_180e_kinetics400_rgb_20200618-3fce5629.pth
#/home/administrator/Z/Algorithms/mmaction2/demo/label_map_k400.txt







