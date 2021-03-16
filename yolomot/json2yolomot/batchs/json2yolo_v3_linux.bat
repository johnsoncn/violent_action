#1 - Extract Images and labels from dataset
"/home/nmc_costa/miniconda3/envs/violent_action/bin/python" "/mnt/c/Users/nmc_costa/google_drive/projects/bosch_P19/research/python_ws/violent_action/yolo/json2yolo/json2yolo.py" \
 --datasets_root_dir="/mnt/d/external_datasets/" \
 --json_dir="/mnt/d/external_datasets/MOLA/annotations/splitann_mola_fix_equal/split_mola_fix_equal/" \
 --outdir="/mnt/d/external_datasets/YOLOTESTS/yoloformat/split_mola_fix_equal_100/" \
 --img_number="100" \
 --copy_images="1" \
 --only_labels="1"
#sleep 2s

