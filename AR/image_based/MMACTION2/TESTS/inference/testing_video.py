import torch
import argparse
import os
from mmaction.apis import init_recognizer, inference_recognizer

parser = argparse.ArgumentParser(description="parsing...")
parser.add_argument("--root", type=str, default="/home/administrator/Z/Algorithms/mmaction2/", help="mmaction2 root")
args = parser.parse_args()

config_file = os.path.join(args.root, 'configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py')
config_file = os.path.join(args.root, config_file)
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = os.path.join(args.root, 'checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth')

# assign the desired device.
device = 'cuda:0' # or 'cpu'
device = torch.device(device)

# build the model from a config file and a checkpoint file
model = init_recognizer(config_file, checkpoint_file, device=device)

# test a single video and show the result:
video = os.path.join(args.root,'demo/demo.mp4')
labels = os.path.join(args.root,'demo/label_map_k400.txt')
results = inference_recognizer(model, video, labels)


# show the results
print(f'The top-5 labels with corresponding scores are:')
for result in results:
    print(f'{result[0]}: ', result[1])
