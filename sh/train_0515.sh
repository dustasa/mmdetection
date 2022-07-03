python -m torch.distributed.launch --nproc_per_node=1 tools/train.py configs/retinanet/retinanet_replknet31B_1Kpretrain_fpn_1x_mini_coco.py --launcher pytorch --options model.backbone.pretrained=checkpoints/RepLKNet-31B_ImageNet-1K_FCOS_COCO.pth