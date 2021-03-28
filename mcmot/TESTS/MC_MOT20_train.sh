"/home/administrator/miniconda3/envs/violent_action/bin/python" "/home/administrator/Z/Algorithms/MCMOT/src/train.py" \
mot \
--dataset="MOT20" \
--data_dir="/home/administrator/Desktop/DATASET/mot/MOT20" \
--data_cfg="/home/administrator/Desktop/DATASET/mot/MOT20/MOT20_cfg.json" \
--exp_id="MC_MOT20_cocodla_1" \
--load_model="/home/administrator/Z/Algorithms/MCMOT/models/ctdet_coco_dla_2x.pth" \
--gpus="0" \
--batch_size="8" \
--num_epochs="10" \
--num_workers="16" 
#--lr_step="50" \
#--trainval \
#--val_intervals="10" \
#--load_model="/home/administrator/Z/Algorithms/FairMOT/models/ctdet_coco_dla_2x.pth" \
#--K 500??
