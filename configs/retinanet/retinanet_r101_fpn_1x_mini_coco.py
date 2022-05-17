_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/mini_coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
