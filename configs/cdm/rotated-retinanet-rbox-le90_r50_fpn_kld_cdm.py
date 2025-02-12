_base_ = '../cdm/rotated-retinanet-rbox-le90_r50_fpn.py'

angle_version = 'le90'
model = dict(
    bbox_head=dict(
        anchor_generator=dict(angle_version=None),
        type='CDAngleBranchRetinaHead',
        use_normalized_angle_feat=True,
        angle_coder=dict(
            type='CDMCoder',
            angle_version=angle_version,
        ),
        reg_decoded_bbox=True,
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            _delete_=True,
            type='GDLoss',
            loss_type='kld',
            fun='log1p',
            tau=1,
            sqrt=False,
            loss_weight=1),
        loss_angle=dict(type='mmdet.L1Loss', loss_weight=0.2)))