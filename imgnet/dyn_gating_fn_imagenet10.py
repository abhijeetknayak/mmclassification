dataset_type = 'ImageNet10'
img_norm_cfg = dict(
    mean=[0.4878, 0.4746, 0.4434], std=[0.2738, 0.2643, 0.2922], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=64),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='Normalize',
        mean=[0.4878, 0.4746, 0.4434],
        std=[0.2738, 0.2643, 0.2922],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=64),
    dict(
        type='Normalize',
        mean=[0.4878, 0.4746, 0.4434],
        std=[0.2738, 0.2643, 0.2922],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type='ImageNet10',
        data_prefix='data/imagenet-10/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', size=64),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean=[0.4878, 0.4746, 0.4434],
                std=[0.2738, 0.2643, 0.2922],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='ImageNet10',
        data_prefix='data/imagenet-10/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', size=64),
            dict(
                type='Normalize',
                mean=[0.4878, 0.4746, 0.4434],
                std=[0.2738, 0.2643, 0.2922],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        test_mode=True),
    test=dict(
        type='ImageNet10',
        data_prefix='data/imagenet-10/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', size=64),
            dict(
                type='Normalize',
                mean=[0.4878, 0.4746, 0.4434],
                std=[0.2738, 0.2643, 0.2922],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        test_mode=True))
model = dict(
    type='ImageClassifier',
    backbone=dict(type='GatingFnNewPretrained', is_train=True, depth=18),
    neck=dict(type='GAP', img_size=1),
    head=dict(
        type='GatingFnLinearHead',
        num_classes=10,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=10, gamma=0.9)
runner = dict(type='EpochBasedRunner', max_epochs=200)
checkpoint_config = dict(interval=5)
evaluation = dict(
    interval=5,
    save_best='auto',
    metric='accuracy',
    metric_options=dict(topk=(1, 5)))
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = 'imgnet'
gpu_ids = [0]
