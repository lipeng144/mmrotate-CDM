# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import xml.etree.ElementTree as ET
from collections import OrderedDict
import os
import mmcv
import numpy as np
from mmengine.logging import print_log
from PIL import Image
from mmengine.fileio import get, get_local_path, list_from_file
from mmrotate.datasets.transforms.rotatedtrans import  obb2poly_np, poly2obb_np
from mmrotate.registry import DATASETS
from mmengine.dataset import BaseDataset

@DATASETS.register_module()
class RSDDDataset(BaseDataset):
    """RSDD dataset for detection.

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        img_subdir (str): Subdir where images are stored. Default: JPEGImages.
        ann_subdir (str): Subdir where annotations are. Default: Annotations.
        classwise (bool): Whether to use all classes or only ship.
        version (str, optional): Angle representations. Defaults to 'oc'.
    """

    CLASSES = None
    HRSC_CLASS = ('ship', )

    PALETTE = [
        (0, 255, 0),
    ]


    def __init__(self,
                 ann_file,
                 pipeline,
                 img_subdir='JPEGImages',
                 ann_subdir='Annotations',
                 classwise=False,
                 version='le90',
                 difficult = 100,
                 **kwargs):
        self.img_subdir = img_subdir
        self.ann_subdir = ann_subdir
        self.classwise = classwise
        self.version = version
        self.difficult = difficult
        if self.classwise:
            RSDDDataset.PALETTE = RSDDDataset.CLASSWISE_PALETTE
            RSDDDataset.CLASSES = self.HRSC_CLASSES
            self.catid2label = {
                ('1' + '0' * 6 + cls_id): i
                for i, cls_id in enumerate(self.HRSC_CLASSES_ID)
            }
        else:
            RSDDDataset.CLASSES = self.HRSC_CLASS
        # self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        super(RSDDDataset, self).__init__(ann_file, pipeline, **kwargs)

    def load_annotations(self, ann_file):
        """Load annotation from XML style ann_file.

        Args:
            ann_file (str): Path of Imageset file.

        Returns:
            list[dict]: Annotation info from XML file.
        """

        data_infos = []
        img_ids = list_from_file(ann_file)
        for img_id in img_ids:
            data_info = {}

            filename = osp.join(self.img_subdir, f'{img_id}.jpg')
            data_info['filename'] = f'{img_id}.jpg'
            xml_path = osp.join(self.ann_subdir,
                                f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()

            width = int(root.find('size/width').text)
            height = int(root.find('size/height').text)
            depth = int(root.find('size/depth').text)
            if width is None or height is None:
                img = Image.open(filename)
                width, height = img.size
                depth = len(img.split())
            data_info['width'] = width
            data_info['height'] = height
            data_info['depth'] = depth
            data_info['ann'] = {}
            gt_bboxes = []
            gt_labels = []
            gt_polygons = []
            gt_difficult = []


            for obj in root.findall('object'):
                obj_difficult = int(obj.find('difficult').text)
                if obj_difficult > self.difficult:
                    pass

                if self.classwise:
                    class_id = obj.find('Class_ID').text
                    label = self.catid2label.get(class_id)
                    if label is None:
                        continue
                else:
                    label = 0

                # Add an extra score to use obb2poly_np
                # 这里obb2poly_np函数调用时输入参数为(cx,cy,w,h,thrta), 但数据集标注时采用长边定义法，所以第三个参数w为长边
                # 详细分析参见博客https://editor.csdn.net/md/?articleId=129341850
                w = float(obj.find('robndbox/w').text)
                h = float(obj.find('robndbox/h').text)
                bbox = np.array([[
                    float(obj.find('robndbox/cx').text),
                    float(obj.find('robndbox/cy').text),
                    max(w,h),
                    min(w,h),
                    float(obj.find('robndbox/angle').text), 0
                ]],
                                dtype=np.float32)

                polygon = obb2poly_np(bbox, 'le90')[0, :-1].astype(np.float32) # 这里的le90是指读取标注时数据集本身采用的旋转框标注格式
                if self.version != 'le90': # 这里的角度version是指在程序中进行处理的版本
                    bbox = np.array(
                        poly2obb_np(polygon, self.version), dtype=np.float32)
                else:
                    bbox = bbox[0, :-1]

                gt_bboxes.append(bbox)
                gt_labels.append(label)
                gt_polygons.append(polygon)
                gt_difficult.append(obj_difficult)

            if gt_bboxes:
                data_info['ann']['bboxes'] = np.array(
                    gt_bboxes, dtype=np.float32)
                data_info['ann']['labels'] = np.array(
                    gt_labels, dtype=np.int64)
                data_info['ann']['polygons'] = np.array(
                    gt_polygons, dtype=np.float32)
                data_info['ann']['difficult'] = np.array(
                    gt_difficult, dtype=np.int16)
            else:
                data_info['ann']['bboxes'] = np.zeros((0, 5), dtype=np.float32)
                data_info['ann']['labels'] = np.array([], dtype=np.int64)
                data_info['ann']['polygons'] = np.zeros((0, 8),
                                                        dtype=np.float32)
                data_info['ann']['difficult'] = np.array([], dtype=np.int16)

            data_infos.append(data_info)

        return data_infos

    def _filter_imgs(self):
        """Filter images without ground truths."""
        valid_inds = []
        for i, data_info in enumerate(self.data_infos):
            if (not self.filter_empty_gt
                    or data_info['ann']['labels'].size > 0):
                valid_inds.append(i)
        return valid_inds
