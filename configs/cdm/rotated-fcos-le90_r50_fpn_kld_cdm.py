_base_ = '../cdm/rotated-fcos-le90_le90-r50_fpn.py'

model = dict(
    bbox_head=dict(
            angle_coder=dict(
            type='CDMCoder',
            angle_version='le90',
        ),
        loss_bbox=dict(
            _delete_=True,
            type='GDLoss_v1',
            loss_type='kld',
            fun='log1p',
            tau=1,
            loss_weight=5.0),
        loss_angle=dict(_delete_=True,type='mmdet.L1Loss', loss_weight=0.2)))
