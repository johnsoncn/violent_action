#1 - Extract Images and labels from dataset
"/home/nmc_costa/miniconda3/envs/violent_action/bin/python" "/mnt/c/Users/nmc_costa/google_drive/projects/bosch_P19/research/python_ws/violent_action/fairmot/json2fairmot/json2fairmot.py" \
 --datasets_root_dir="/mnt/d/external_datasets/" \
 --json_dir="/mnt/d/external_datasets/TAO/TAO_DIR/annotations/" \
 --outdir="/mnt/d/external_datasets/FAIRMOTESTS/fairmotformat/TAO_TEST/" \
 --img_number="300" \
 --copy_images="1" \
 --only_labels="1"
#sleep 3s
#2 - Extract Images and labels from dataset
