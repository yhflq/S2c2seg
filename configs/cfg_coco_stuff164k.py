# COCO-Stuff164k
_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_coco_stuff.txt',
    # prob_thd = 0
    x_options=dict(
        X_PAPER_PIPELINE=True,
        X_PAPER_TAU=0.3,
        X_PAPER_LAM=0.15,
    ),
)

# dataset settings
dataset_type = 'COCOStuffDataset'
data_root = 'data/COCOStuff164k'

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
            img_path='images/val2017', seg_map_path='annotations/val2017'),
        pipeline=test_pipeline))
