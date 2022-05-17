_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/mini_coco_detection.py',
    '../_base_/schedules/schedule_0.5x.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        depth=152,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet152')))
# optimizer
# optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
