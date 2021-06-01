mkdir $PWD/"results"
for i in /home/administrator/Z/Work/EASYRIDE/P19/NC/SAMPLES/MOLA_2P_BACK_MP4_SMALL_5V/*.mp4;
  do name="$(basename -s .avi $i)"
  echo "$name"
"/home/administrator/miniconda3/envs/mmaction2/bin/python" "/home/administrator/Z/Algorithms/mmaction2/demo/long_video_demo.py" \
/home/administrator/Z/Algorithms/mmaction2/configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
"$i" \
/home/administrator/Z/Work/EASYRIDE/P19/NC/mmaction2/INFERENCES/I3/label_map_k400_violent.txt \
$PWD/"results/${name}_longdemo.mp4" \
--input-step 3 \
--threshold 0.01 
done

#https://github.com/open-mmlab/mmaction2/tree/master/demo#video-demo


