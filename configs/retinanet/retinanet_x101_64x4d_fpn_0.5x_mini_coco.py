_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/mini_coco_detection.py',
    '../_base_/schedules/schedule_0.5x.py', '../_base_/default_runtime.py'
]
model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')))
