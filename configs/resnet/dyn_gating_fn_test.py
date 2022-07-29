# dataset settings
dataset_type = 'CIFAR10'
img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=False)
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type, data_prefix='data/cifar10',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/cifar10',
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_prefix='data/cifar10',
        pipeline=test_pipeline,
        test_mode=True))

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='GatingFnNetTest',
        is_train=False,
        depth=18),
    neck=dict(type='GAP', img_size=1),
    head=dict(
        type='GatingFnLinearHead',
        num_classes=10,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

# The configuration file used to build the optimizer, support all optimizers in PyTorch.
optimizer = dict(type='SGD',         # Optimizer type
                lr=0.05,              # Learning rate of optimizers, see detail usages of the parameters in the documentation of PyTorch
                momentum=0.9,        # Momentum
                weight_decay=0.0001) # Weight decay of SGD
# Config used to build the optimizer hook, refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/optimizer.py#L8 for implementation details.
optimizer_config = dict(grad_clip=None)  # Most of the methods do not use gradient clip
# Learning rate scheduler config used to register LrUpdater hook
lr_config = dict(policy='step',          # The policy of scheduler, also support CosineAnnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9.
                 step=[30, 60, 90])      # Steps to decay the learning rate
runner = dict(type='EpochBasedRunner',   # Type of runner to use (i.e. IterBasedRunner or EpochBasedRunner)
            max_epochs=100)    # Runner that runs the workflow in total max_epochs. For IterBasedRunner use `max_iters`

# Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
checkpoint_config = dict(interval=1)    # The save interval is 1
# config to register logger hook
log_config = dict(
    interval=100,                       # Interval to print the log
    hooks=[
        dict(type='TextLoggerHook'),           # The Tensorboard logger is also supported
        # dict(type='TensorboardLoggerHook')
    ])

dist_params = dict(backend='nccl')   # Parameters to setup distributed training, the port can also be set.
log_level = 'INFO'             # The output level of the log.
resume_from = None             # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved.
workflow = [('train', 1)]      # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once.
work_dir = ''          # Directory to save the model checkpoints and logs for the current experiments.
load_from = None