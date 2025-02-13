# dataset settings
dataset_type = 'mmdet.CocoDataset'
data_root = 'D:/pythondata/mmrotate-1.x/tools/data/rsdd/'
backend_args = None

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(
        type='mmdet.LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='ConvertMask2BoxType', box_type='qbox'),
    dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
    # avoid bboxes being resized
    dict(
        type='mmdet.LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='ConvertMask2BoxType', box_type='qbox'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'instances'))
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

metainfo = dict(classes=('ship', ))

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='ImageSets/train.json',
        data_prefix=dict(img='JPEGImages/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='ImageSets/test.json',
        data_prefix=dict(img='JPEGImages/'),
        test_mode=True,
        pipeline=val_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

#val_evaluator = dict(type='RotatedCocoMetric', metric='bbox',outfile_prefix='D:/pythondata/mmrotate-1.x/tools/work_dirs/rsdd-retinanet/rotated-retinanet-rbox-le90_r50_fpn_1x_rsdd/20240827_154026/testresult')
val_evaluator = dict(
    type='DOTAMetric', metric='mAP', iou_thrs=0.2, predict_box_type='qbox')
test_evaluator = val_evaluator
