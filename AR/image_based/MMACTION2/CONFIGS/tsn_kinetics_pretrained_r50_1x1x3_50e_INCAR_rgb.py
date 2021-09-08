# CONFIG SETTINGS ACTION RECOGNITION
# (ALL THE DEFAULT SESTTINGS can be overwritten)
#info: 
	#1-learn about configs, https://github.com/open-mmlab/mmaction2/blob/master/docs/tutorials/1_config.md
	#2-finetuning models from base, https://github.com/open-mmlab/mmaction2/blob/master/docs/tutorials/2_finetune.md

# NAME (using FILE NAMING CONVENTION): 
	#{model}_[model setting]_{backbone}_[misc]_{data setting}_[gpu x batch_per_gpu]_{schedule}_{dataset}_{modality}
	#tsn_r50_1x1x3_50e_INCAR_rgb

# example: tsn_r50_1x1x3_100e_kinetics400_rgb.py
# WARNING: to use default paths this script needs to be in mmaction2/configs/...

# mmaction2 dir
mmadir="/home/administrator/Z/Algorithms/mmaction2/"

# base(default) settings
baseroot=mmadir+'configs/' #default: "../../"
_base_ = [
    baseroot+'_base_/default_runtime.py' #_base_ runtime settings 
]     
# baseroot+'_base_/models/tsn_r50.py', #_base_ model settings
# baseroot+'_base_/schedules/sgd_100e.py', #_base_ optimizer, learning policy and epochs


# model settings
model = dict(  # Config of the model
    type='Recognizer2D',  # Type of the recognizer
    backbone=dict(  # Dict for backbone
        type='ResNet',  # Name of the backbone
        pretrained='torchvision://resnet50',  # The url/site of the pretrained model
        depth=50,  # Depth of ResNet model
        norm_eval=False),  # Whether to set BN layers to eval mode when training
    cls_head=dict(  # Dict for classification head
        type='TSNHead',  # Name of classification head
        num_classes=2,  # Number of classes to be classified. #CHANGED TO 2 VIOLENT/NONVIOLENT
        in_channels=2048,  # The input channels of classification head.
        spatial_type='avg',  # Type of pooling in spatial dimension
        consensus=dict(type='AvgConsensus', dim=1),  # Config of consensus module
        dropout_ratio=0.4,  # Probability in dropout layer
        init_std=0.01), # Std value for linear layer initiation
        # model training and testing settings
        train_cfg=None,  # Config of training hyperparameters for TSN
        test_cfg=dict(average_clips=None))  # Config for testing hyperparameters for TSN.

# dataset settings
root=mmadir+"data/" #default: "data/"
dataset="INCAR"
filename_tmpl='img_{:05}.png' #filename template of saved frames
dataset_type = 'RawframeDataset'
data_root = root+dataset+'/rawframes'
data_root_val = root+dataset+'/rawframes'
ann_file_train = root+dataset+'/'+dataset+'_train_rawframes.txt'
ann_file_val = root+dataset+'/'+dataset+'_val_rawframes.txt'
ann_file_test = root+dataset+'/'+dataset+'_test_rawframes.txt'
img_norm_cfg = dict( # Config of image normalization used in data pipeline
    mean=[123.675, 116.28, 103.53], # Mean values of different channels to normalize
    std=[58.395, 57.12, 57.375], # Std values of different channels to normalize
    to_bgr=False) # Whether to convert channels from RGB to BGR
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=3),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=3,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=25,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='TenCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict( # Config of data
    videos_per_gpu=32, # Batch size of each single GPU
    workers_per_gpu=4, # Workers to pre-fetch data for each single GPU
    #train_dataloader=dict(  # Additional config of train dataloader
	#    drop_last=True),  # Whether to drop out the last batch of data in training
	#val_dataloader=dict(  # Additional config of validation dataloader
	#    videos_per_gpu=1),  # Batch size of each single GPU during evaluation
	#test_dataloader=dict(  # Additional config of test dataloader
	#    videos_per_gpu=2),  # Batch size of each single GPU during testing
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        filename_tmpl=filename_tmpl,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        filename_tmpl=filename_tmpl,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        filename_tmpl=filename_tmpl,
        pipeline=test_pipeline))


# optimizer (training schedule settings)
optimizer = dict(
    # Config used to build optimizer, support (1). All the optimizers in PyTorch
    # whose arguments are also the same as those in PyTorch. (2). Custom optimizers
    # which are built on `constructor`, referring to "tutorials/5_new_modules.md"
    # for implementation.
    type='SGD',
    lr=0.01,  # this lr is used for 8 gpus = 0.01 # single-gpu = 0.01 / 8 
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[40, 80]) #change step rom  [40, 80]
total_epochs = 50 # change from 100 to 50
checkpoint_config = dict(interval=5) # Interval to save checkpoint
evaluation = dict(  # Config of evaluation during training
    interval=5,  # Interval to perform evaluation
    metrics=['top_k_accuracy', 'mean_class_accuracy'],  # Metrics to be performed
    metric_options=dict(top_k_accuracy=dict(topk=(1, 3))), # Set top-k accuracy to 1 and 3 during validation
    save_best='top_k_accuracy')  # set `top_k_accuracy` as key indicator to save best checkpoint
eval_config = dict(
    metric_options=dict(top_k_accuracy=dict(topk=(1, 3)))) # Set top-k accuracy to 1 and 3 during testing. You can also use `--eval top_k_accuracy` to assign evaluation metrics
log_config = dict(  # Config to register logger hook
    interval=50,  # Interval to print the log
    hooks=[  # Hooks to be implemented during training
        dict(type='TextLoggerHook'),  # The logger used to record the training process
        # dict(type='TensorboardLoggerHook'),  # The Tensorboard logger is also supported
    ])


# runtime settings
dist_params = dict(backend='nccl')  # Parameters to setup distributed training, the port can also be set
log_level = 'INFO'  # The level of logging
work_dir_root= '/home/administrator/Z/Work/EASYRIDE/P19/NC/mmaction2/TESTS/train/' #defualt: './'
work_dir = work_dir_root+'work_dirs/tsn_kinetics_pretrained_r50_1x1x3_50e_INCAR_rgb/' # Directory to save the model 
load_from = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/mmaction-v1/recognition/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'  # model path can be found in model zoo  # load models as a pre-trained model from a given path. This will not resume training
resume_from = None  # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved
workflow = [('train', 1)]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once
output_config = dict(  # Config of localization ouput
    out=f'{work_dir}/results.json',  # Path to output file
    output_format='json')  # File format of output file

# single-gpu training (WARNING)
# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
#data.videos_per_gpu = cfg.data.videos_per_gpu // 16
#optimizer.lr = cfg.optimizer.lr / 8 / 16
#total_epochs = 30


