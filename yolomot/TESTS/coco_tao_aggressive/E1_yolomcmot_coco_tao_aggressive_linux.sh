"/home/administrator/miniconda3/envs/violent_action/bin/python" \
"/home/administrator/Z/Algorithms/YOLOV4_MCMOT/train.py" \
 --img="512" \
 --batch="56" \
 --epochs="10" \
 --data="/home/administrator/Desktop/EXPERIMENTS/yolomot/ENSAIOS/coco_tao_aggressive/coco_tao_aggressive.data" \
 --cfg="/home/administrator/Desktop/EXPERIMENTS/yolomot/ENSAIOS/coco_tao_aggressive/yolov4-tiny.cfg" \
 --name="E1_coco_tao_aggressive" \
 --device 0


#YOLOV4_MCMOT PARAMETERS
#train.py 
#SAVE results to YOLOV4_MCMOT/weights
#--img="512" <-default=[384, 832, 768], help='[min_train, max-train, test]'<-if img="512"->[512, 512, 512]
#--hyp="/home/administrator/Desktop/YAMLS/hyp.scratch.yaml" \ <-#DONT'T EXISTS hyp defined in train.py
# --workers="16" \ <-#DONT'T EXISTS is defined as nw=8 in train.py
# --data="*.data path" <-txt file path
# --cfg="" \ <-'cfg/yolov4-tiny-3l_no_group_id_no_upsample.cfg' ->copy and change "classes=" parameter
# --evolve \ <-evolve is not prepared in train.py ->for _ in range(1):  # generations to evolve
# --weights="" \ <-'./weights/track_last.weights'
# --task="" \ :     
	# Set 3 task mode: pure_detect | detect | track(default)
   	# pure detect means the dataset do not contains ID info.
	# detect means the dataset contains ID info, but do not load for training. (i.e. do detection in tracking)
	# track means the dataset contains both detection and ID info, use both for training. (i.e. detect & reid)
#'--auto-weight', type=bool, default=False, help='Whether use auto weight tuning'

