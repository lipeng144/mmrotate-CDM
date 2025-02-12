# %%
import os
import mmcv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from mmrotate.utils import register_all_modules
from mmengine.fileio import load
from mmengine.config import Config
#from .datasets import build_dataset
from mmdet.registry import DATASETS

def getPrecisions(config_file, result_file, metric="bbox"):
    """plot precison-recall curve based on testing results of pkl file.

        Args:
            config_file (list[list | tuple]): config file path.
            result_file (str): pkl file of testing results path.
            metric (str): Metrics to be evaluated. Options are
                'bbox', 'segm'.
    """

    cfg = Config.fromfile(config_file)
    #turn on test mode of dataset
    # if isinstance(cfg.data.test, dict):
    #     cfg.data.test.test_mode = True
    # elif isinstance(cfg.data.test, list):
    #     for ds_cfg in cfg.data.test:
    #         ds_cfg.test_mode = True

    # build dataset
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    # load result file in pkl format
    pkl_results = load(result_file)
    # convert pkl file (list[list | tuple | ndarray]) to json
    json_results, _ = dataset.format_results(pkl_results)
    # initialize COCO instance
    coco = COCO(annotation_file=cfg.test_dataloader.dataset.ann_file)
    coco_gt = coco
    coco_dt = coco_gt.loadRes(json_results[metric])
    # initialize COCOeval instance
    coco_eval = COCOeval(coco_gt, coco_dt, metric)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    '''
    precisions[T, R, K, A, M]
    T: iou thresholds [0.5 : 0.05 : 0.95], idx from 0 to 9
    R: recall thresholds [0 : 0.01 : 1], idx from 0 to 100
    K: category, idx from 0 to ...
    A: area range, (all, small, medium, large), idx from 0 to 3
    M: max dets, (1, 10, 100), idx from 0 to 2
    '''
    return coco_eval.eval["precision"], coco_eval.eval["recall"]


def PR(config, result, out, thr=0.5):
    """Export PR Excel data

        Args:
            config_file (list[list | tuple]): config file path.
            result_file (str): pkl file of testing results path.
            out (str): path of excel file
            thr(float): output PR Threshold. Optional range: {-1, [0.5, 0.95]}
                If thr == -1: Threshold is 0.5-0.95
    """
    '''
    precisions[T, R, K, A, M]
    T: iou thresholds [0.5 : 0.05 : 0.95], idx from 0 to 9
    R: recall thresholds [0 : 0.01 : 1], idx from 0 to 100
    K: category, idx from 0 to ...
    A: area range, (all, small, medium, large), idx from 0 to 3
    M: max dets, (1, 10, 100), idx from 0 to 2
    '''

    root = 'D:/pythondata/mmrotate-1.x/tools/work_dirs/pr-curve'
    fileList = os.listdir(root)
    linestyles = ['-', '-', '-.', ':', '-', '--']
    markers = [".", "^", " ", " ", " ", " "]

    for i, file in enumerate(fileList):
        precisions, recall = getPrecisions(
            os.path.join(root, file, 'config.py'),
            os.path.join(root, file, 'result.pkl'))

        # precisions, recall = getPrecisions(config, result)

        x = np.arange(0.0, 1.01, 0.01)
        precision = np.mean(precisions[0, :, :, 0, -1], 1)  # shape: (101,)

        plt.plot(x, precision, label=file, linestyle=linestyles[i], marker=markers[i], markevery=5, color='black')

    # # plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.01)
    # plt.grid(True)
    plt.legend(loc="lower left")
    plt.title('COCO mAP@0.5 PR Curve')
    plt.show()


if __name__ == '__main__':
   register_all_modules()
   PR('rsdd-retinanet/rotated-retinanet-rbox-le90_r50_fpn_1x_rsdd/rotated-retinanet-rbox-le90_r50_fpn_1x_rsdd.py', 'rsdd-retinanet/rotated-retinanet-rbox-le90_r50_fpn_1x_rsdd/20240827_154026/','ours')
