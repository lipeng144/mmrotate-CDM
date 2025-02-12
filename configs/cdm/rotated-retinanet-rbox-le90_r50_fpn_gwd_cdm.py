_base_ = '../cdm/rotated-retinanet-rbox-le90_r50_fpn.py'

angle_version = 'le90'
model = dict(
    bbox_head=dict(
        anchor_generator=dict(angle_version=None),
        type='CDAngleBranchRetinaHead',
        use_normalized_angle_feat=True,
        angle_coder=dict(
            type='ACMCoder',
            angle_version=angle_version,
            dual_freq=False,
            #base_omega=2
        ),
        reg_decoded_bbox=True,
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GDLoss', loss_type='gwd', loss_weight=5),
        loss_angle=dict(type='mmdet.L1Loss', loss_weight=0.2)))
