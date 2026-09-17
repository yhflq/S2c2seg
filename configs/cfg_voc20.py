# PASCAL VOC20
_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_voc20.txt',
    # prob_thd = 0
    x_options=dict(
        X_PAPER_PIPELINE=True,
        X_PAPER_TAU=0.3,
        X_PAPER_LAM=0.15,
    ),
)

# dataset settings
dataset_type = 'PascalVOC20Dataset'
data_root = 'data/VOC2012'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 336), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/Segmentation/val.txt',
        pipeline=test_pipeline))
