# Cityscapes
_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_city_scapes.txt',
    slide_stride=112,
    slide_crop=224,
    # prob_thd = 0
    x_options=dict(
        X_PAPER_PIPELINE=True,
        X_PAPER_TAU=0.6,
        X_PAPER_LAM=0.08,
    ),
)

# dataset settings
dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 448), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
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
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        pipeline=test_pipeline))
