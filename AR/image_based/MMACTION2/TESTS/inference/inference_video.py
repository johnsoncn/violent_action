import torch
import argparse
import os
from mmaction.apis import init_recognizer, inference_recognizer

#parser
parser = argparse.ArgumentParser(description="mmaction2 inference top 5 results")
parser.add_argument(
    '--config',
    default=('/home/administrator/Z/Algorithms/mmaction2/configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py'),
    help='spatio temporal detection config file path')
parser.add_argument(
    '--checkpoint',
    default=('/home/administrator/Z/Algorithms/mmaction2/checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'),
    help='spatio temporal detection checkpoint file/url')
parser.add_argument(
    '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
parser.add_argument(
	'--video', help='video file/url')
parser.add_argument(
	'--labels', help='dataset labels')
args = parser.parse_args()

#config file
config_file = args.config

# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = args.checkpoint

# assign the desired device.
device = args.device # 'cuda:0' or 'cpu'
device = torch.device(device)

# build the model from a config file and a checkpoint file
model = init_recognizer(config_file, checkpoint_file, device=device)

# test a single video and show the result:
video = args.video
labels = args.labels
results = inference_recognizer(model, video, labels)


# show the results
print(f'The top-5 labels with corresponding scores are:')
for result in results:
    print(f'{result[0]}: ', result[1])
