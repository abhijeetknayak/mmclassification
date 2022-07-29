# dataset settings
dataset_type = 'CustomDataset'
# classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

img_norm_cfg = dict(
    mean=[0.4302, 0.4575, 0.4539], 
    std=[0.2700, 0.2681, 0.2986],
    to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomCrop', size=100, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomCrop', size=64, padding=4),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type, data_prefix='data/caltech/256_ObjectCategories',
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/caltech/256_ObjectCategories',
        classes=classes,
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_prefix='data/caltech/256_ObjectCategories',
        classes=classes,
        pipeline=test_pipeline,
        test_mode=True))

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='GatingFnNet2',
        is_train=True,
        depth=18),
    neck=dict(type='GAP', img_size=2),
    head=dict(
        type='GatingFnLinearHead',
        num_classes=11,
        in_channels=512 * 4,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[50, 75])
runner = dict(type='EpochBasedRunner', max_epochs=120)

checkpoint_config = dict(interval=5)

# "auto" means automatically select the metrics to compare.
# You can also use a specific key like "accuracy_top-1".
evaluation = dict(interval=5, save_best="auto", metric='accuracy', metric_options={'topk': (1, 5)})

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook')
    ]) 

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = 'gating_fn_net_hard_caltech' 