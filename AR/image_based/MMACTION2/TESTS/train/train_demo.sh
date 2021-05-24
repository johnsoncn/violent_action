#Single GPU 
#for multiple you use ./tools/dist_train.sh

"/home/administrator/miniconda3/envs/mmaction2/bin/python" \
"/home/administrator/Z/Algorithms/mmaction2/tools/train.py" \
/home/administrator/Z/Algorithms/mmaction2/configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
--validate \
--work-dir="/home/administrator/Z/Work/EASYRIDE/P19/NC/mmaction2/TESTS" \
--resume-from="/home/administrator/Z/Algorithms/mmaction2/checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth" \ #: Resume from a previous checkpoint file.

#Difference between resume-from and load-from: resume-from loads both the model weights and optimizer status, and the epoch is also inherited from the specified checkpoint. It is usually used for resuming the training process that is interrupted accidentally. load-from only loads the model weights and the training epoch starts from 0. It is usually used for finetuning.

#MMAction2 implements distributed training and non-distributed training, which uses MMDistributedDataParallel and MMDataParallel respectively. Each process keeps an isolated model, data loader, and optimizer. Model parameters are only synchronized once at the beginning. After a forward and backward pass, gradients will be allreduced among all GPUs, and the optimizer will update model parameters. Since the gradients are allreduced, the model parameter stays the same for all processes after the iteration.
