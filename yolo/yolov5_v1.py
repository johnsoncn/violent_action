#!/usr/bin/env python
# coding: utf-8

# # yolov5
# version 1
# 
# author: nmc-costa
# 
# info:
# 

# # Setup
# Go to tutorial_yolov5.ipnb for full setup (like cloning the yolov5 repo, etc...)

# In[1]:


import os
import sys
import torch
from IPython.display import Image, clear_output  # to display images
from IPython import get_ipython

#script dir [in case you use cd]
scriptdir=os.getcwd()

#cuda or cpu version
clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))


# In[5]:


print(scriptdir)


# In[7]:


#yolov5 repo [other ways: %cd yolov5; or pip install yolov5 module as "editable" using a setup.py file]
import yolov5
from yolov5.utils.google_utils import gdrive_download  # to download models/datasets
#Dir
yolov5_dir=os.path.join(scriptdir, 'yolov5')


# ## init vars

# ## input data

# #### Prepare the dataset for yolo format (CHECK OUT tutorial_customdata.ipynb)
# 
# 1. create yaml file (like coco_external) for the dataset
# 
# Yolov5 algorithm parses yaml information: path to dataset(train, validation and test) - YOLO (darknet) format; number of classes and classes names
# 
# Dataset Directory structure:
# - dataset:
#     - annotations:
#         - json file: instances_samefoldername.json
#     - images:
#         - samefoldername
#     - labels:
#         - samefoldername
# 
# 
# 2. check yaml and dataset paths if they are correct
# 
# 3. based on the path and parameters yolov5 creates a torch dataloader (create_dataloader func) from the dataset images and labels
# 

# In[8]:


#YAML Paths
coco_yaml=os.path.join(scriptdir, 'coco_external.yaml')
coco128_yaml=os.path.join(scriptdir, 'coco128_external.yaml')


# #### Dowload coco datasets in YOLOv5 (darknet) format
# Download COCO val 2017 dataset, 1GB, 5000 images, and test model accuracy. (use function gdrive_dowload)

# In[ ]:


# Download COCO val2017
"""
gdrive_download('1Y6Kou6kEB0ZEMCCpJSKStCor4KAReE43','coco2017val.zip')  # val2017 dataset
get_ipython().system('mv ./coco /Volumes/Seagate/p19/external_datasets/coco  # move folder')


# ##### OR
# Download coco128 a small tutorial dataset composed of the first 128 images

# In[3]:


# Download coco128
gdrive_download('1n_oKgR81BJtqk75b00eAjdv03qVCQn2f','coco128.zip')  # coco128 dataset
get_ipython().system('mv ./coco128 /Volumes/Seagate/p19/external_datasets/coco128  # move folder')

"""
# ## preprocessing

# ## processing

# ### Train
# Run the training command below to train `coco128.yaml` for 5 epochs. You can train YOLOv5s from scratch by passing `--cfg yolov5s.yaml --weights ''`, or train from a pretrained checkpoint by passing a matching weights file: `--cfg yolov5s.yaml --weights yolov5s.pt`.

# In[9]:


# Train YOLOv5s on coco128 for 5 epochs
# WINDOWS BUG 1 # UPDATE: yolov5.utils.increment_dir function error in windows path -update try: except: n=0 #UPDATE removed
# WINDOWS BUG 2 : zip() function gives a keyerror
# CODE BUG  : they use 'opt' var declared in if __name__ == '__main__': as a global var : #WARNING can only debug inside script, or you need to recreate the opt as global in your script


get_ipython().run_line_magic('cd', '$yolov5_dir')
get_ipython().system('pwd')
get_ipython().system('ls')
get_ipython().system('python train.py --img 640 --batch 16 --epochs 5 --data $coco128_yaml --cfg ./models/yolov5s.yaml --weights yolov5s.pt')



# ## postprocessing

# ## output data

# ### Test

# In[7]:


# SHELL Ipython: Run YOLOv5x on COCO validation images
"""
get_ipython().run_line_magic('cd', '$yolov5_dir')
get_ipython().system('pwd #current directory')
get_ipython().system('ls #list directory')
get_ipython().system("python test.py --weights yolov5x.pt --data $coco128_yaml --task 'val' --img 640")
"""


"""
#WARNING it cant be used without argparse above, because they use opt. in __main__
Don't install 
data=coco128_yaml
weights='yolov5x.pt'
imgsz=640
from yolo.yolov5 import test
test.test(data,
         weights=weights,
         batch_size=16,
         imgsz=imgsz,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir='',
         merge=False)
"""

# In[5]:




# ## NOMENCLATURE

# In[10]:



# In[ ]:




