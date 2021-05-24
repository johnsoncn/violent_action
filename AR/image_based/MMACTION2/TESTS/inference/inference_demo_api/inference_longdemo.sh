"/home/administrator/miniconda3/envs/mmaction2/bin/python" "/home/administrator/Z/Algorithms/mmaction2/demo/long_video_demo.py" \
/home/administrator/Z/Algorithms/mmaction2/configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
/home/administrator/Z/Work/EASYRIDE/P19/NC/SAMPLES/MOLA_2P_BACK_MP4_SMALL/C3_P2_P1_2.mp4 \
/home/administrator/Z/Algorithms/mmaction2/demo/label_map_k400.txt \
$PWD/C3_P2_P1_2_longdemo.mp4 \
--input-step 3 \
--threshold 0.01 


#https://github.com/open-mmlab/mmaction2/tree/master/demo#video-demo

#input video

# TSN on K400  
#configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
#https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
#https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4 \
#demo/label_map_k400.txt PATH_TO_SAVED_VIDEO --input-step 3 --device cpu --threshold 0.2
  
# I3D on K400
#/home/administrator/Z/Algorithms/mmaction2/configs/recognition/i3d/i3d_r50_video_inference_32x2x1_100e_kinetics400_rgb.py \
#https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_256p_32x2x1_100e_kinetics400_rgb/i3d_r50_256p_32x2x1_100e_kinetics400_rgb_20200801-7d9f44de.pth \

