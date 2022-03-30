#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import queue
import cv2
import torch
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

import slowfast.utils.checkpoint as cu
from slowfast.datasets import cv2_transform
from slowfast.models import build_model
from slowfast.utils import logging
from slowfast.visualization.utils import process_cv2_inputs

logger = logging.get_logger(__name__)

def dfs(point, feat, valid_map, threshold):
    N, T, H, W, C = feat.size()
    point_set = [point]
    point_stack = [point]
    while len(point_stack) > 0:
        w, h = point_stack.pop()
        if h-1 >=0 and valid_map[h-1, w] == 1:
            flag_pos = False
            for n in range(N):
                for t in range(T):
                    for c in range(1, C):
                        if feat[n, t, h-1, w, c] > threshold:
                            flag_pos = True
                            break
                    if flag_pos:
                        break
            if flag_pos:
                point_set.append([w, h-1])
                point_stack.append([w, h-1])
                valid_map[h-1, w] = 0
        
        if h+1 < H and valid_map[h+1, w] == 1:
            flag_pos = False
            for n in range(N):
                for t in range(T):
                    for c in range(1, C):
                        if feat[n, t, h+1, w, c] > threshold:
                            flag_pos = True
                            break
                    if flag_pos:
                        break
            if flag_pos:
                point_set.append([w, h+1])
                point_stack.append([w, h+1])
                valid_map[h+1, w] = 0
        
        if w-1 >=0 and valid_map[h, w-1] == 1:
            flag_pos = False
            for n in range(N):
                for t in range(T):
                    for c in range(1, C):
                        if feat[n, t, h, w-1, c] > threshold:
                            flag_pos = True
                            break
                    if flag_pos:
                        break
            if flag_pos:
                point_set.append([w-1, h])
                point_stack.append([w-1, h])
                valid_map[h, w-1] = 0
        
        if w+1 < W and valid_map[h, w+1] == 1:
            flag_pos = False
            for n in range(N):
                for t in range(T):
                    for c in range(1, C):
                        if feat[n, t, h, w+1, c] > threshold:
                            flag_pos = True
                            break
                    if flag_pos:
                        break
            if flag_pos:
                point_set.append([w+1, h])
                point_stack.append([w+1, h])
                valid_map[h, w+1] = 0
    return point_set, valid_map
        
# index = 0
def detect_result_decode(feat):
    # global index
    # index += 1
    # cv2.imwrite('/home/haoren/repo/slowfast/solution/raw/debug/preds/feat_max' + str(index) + '.jpg', (feat[:, :, :, :, 1]*255).mean(dim=0).max(dim=0)[0].detach().cpu().numpy())
    threshold = 0.4
    N, T, H, W, C = feat.size()
    point_sets = []
    valid_map = torch.ones((H, W))
    bboxes = None
    preds = None
    for n in range(N):
        for h in range(H):
            for w in range(W):
                flag_pos = False
                for t in range(T):
                    for c in range(1, C):
                        if feat[n, t, h, w, c] > threshold:
                            flag_pos = True
                            break
                    if flag_pos:
                        break
                if flag_pos and valid_map[h, w] == 1:
                    valid_map[h, w] = 0
                    point_set, valid_map = dfs([w, h], feat, valid_map, threshold)
                    point_sets.append(point_set)
    for point_set in point_sets:
        if len(point_set) < 5:
            continue
        point_set_np = np.array(point_set)
        x_min = int(min(point_set_np[:, 0]) * 8)
        x_max = int(max(point_set_np[:, 0]) * 8)
        y_min = int(min(point_set_np[:, 1]) * 8)
        y_max = int(max(point_set_np[:, 1]) * 8)
        mean_score = torch.zeros((C), device=feat.device)
        for point in point_set:
            # mean_score = mean_score + feat[:, :, point[1], point[0], :].mean(dim=[1, 0])
            mean_score_cur = feat[:, :, point[1], point[0], :].mean(dim=0).max(dim=0)[0]
            mean_score_cur[0] = 1 - mean_score_cur[1:].sum()
            mean_score = mean_score + mean_score_cur
        mean_score = mean_score / len(point_set)
        if preds is None:
            preds = mean_score.unsqueeze(0)
        else:
            preds = torch.cat([preds, mean_score.unsqueeze(0)], dim=0)
        if bboxes is None:
            bboxes = torch.tensor([[preds[-1].max(), x_min, y_min, x_max, y_max]])
        else:
            bboxes = torch.cat([bboxes, torch.tensor([[preds[-1].max(), x_min, y_min, x_max, y_max]])], dim=0)
    if preds is None:
        preds = torch.zeros((N, C))
        preds[:, 0] = 1
    return bboxes, preds



class Predictor:
    """
    Action Predictor for action recognition.
    """

    def __init__(self, cfg, gpu_id=None):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
            gpu_id (Optional[int]): GPU id.
        """
        if cfg.NUM_GPUS:
            self.gpu_id = (
                torch.cuda.current_device() if gpu_id is None else gpu_id
            )

        # Build the video model and print model statistics.
        self.model = build_model(cfg, gpu_id=gpu_id)
        self.model.eval()
        self.cfg = cfg

        if cfg.DETECTION.ENABLE:
            self.object_detector = Detectron2Predictor(cfg, gpu_id=self.gpu_id)

        logger.info("Start loading model weights.")
        cu.load_test_checkpoint(cfg, self.model)
        logger.info("Finish loading model weights")

    def __call__(self, task):
        """
        Returns the prediction results for the current task.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames, boxes)
        Returns:
            task (TaskInfo object): the same task info object but filled with
                prediction values (a tensor) and the corresponding boxes for
                action detection task.
        """
        if self.cfg.DETECTION.ENABLE:
            task = self.object_detector(task)

        frames, bboxes = task.frames, task.bboxes
        if bboxes is not None:
            bboxes = cv2_transform.scale_boxes(
                self.cfg.DATA.TEST_CROP_SIZE,
                bboxes,
                task.img_height,
                task.img_width,
            )
        if self.cfg.DEMO.INPUT_FORMAT == "BGR":
            frames = [
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames
            ]

        frames = [
            cv2_transform.scale(self.cfg.DATA.TEST_CROP_SIZE, frame)
            for frame in frames
        ]
        inputs = process_cv2_inputs(frames, self.cfg)
        if bboxes is not None:
            index_pad = torch.full(
                size=(bboxes.shape[0], 1),
                fill_value=float(0),
                device=bboxes.device,
            )

            # Pad frame index for each box.
            bboxes = torch.cat([index_pad, bboxes], axis=1)
        if self.cfg.NUM_GPUS > 0:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(
                        device=torch.device(self.gpu_id), non_blocking=True
                    )
            else:
                inputs = inputs.cuda(
                    device=torch.device(self.gpu_id), non_blocking=True
                )
        # import time
        # start = time.time()
        if self.cfg.DETECTION.ENABLE and not bboxes.shape[0]:
            preds = torch.tensor([])
        else:
            preds = self.model(inputs, bboxes)
        # print('model time costum', time.time() - start)
        
        # for t in range(preds.shape[1]):
        #     cv2.imwrite('/home/haoren/repo/slowfast/solution/raw/debug/preds/frame_' + str(t) + '.jpg', preds[0, t, :, :, 1].detach().cpu().numpy()*255)
        #     cv2.imwrite('/home/haoren/repo/slowfast/solution/raw/debug/preds/pic_' + str(t) + '.jpg', inputs[1][0, :, t, :, :].permute(1, 2, 0).detach().cpu().numpy()*127)
        
        # for action detect
        bboxes, preds = detect_result_decode(preds)
        # print('total time costum', time.time() - start)

        if self.cfg.NUM_GPUS:
            preds = preds.cpu()
            if bboxes is not None:
                bboxes = bboxes.detach().cpu()

        preds = preds.detach()
        task.add_action_preds(preds)
        if bboxes is not None:
            task.add_bboxes(bboxes[:, 1:])

        return task


class ActionPredictor:
    """
    Synchronous Action Prediction and Visualization pipeline with AsyncVis.
    """

    def __init__(self, cfg, async_vis=None, gpu_id=None):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
            async_vis (AsyncVis object): asynchronous visualizer.
            gpu_id (Optional[int]): GPU id.
        """
        self.predictor = Predictor(cfg=cfg, gpu_id=gpu_id)
        self.async_vis = async_vis

    def put(self, task):
        """
        Make prediction and put the results in `async_vis` task queue.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames, boxes)
        """
        task = self.predictor(task)
        self.async_vis.get_indices_ls.append(task.id)
        self.async_vis.put(task)

    def get(self):
        """
        Get the visualized clips if any.
        """
        try:
            task = self.async_vis.get()
        except (queue.Empty, IndexError):
            raise IndexError("Results are not available yet.")

        return task


class Detectron2Predictor:
    """
    Wrapper around Detectron2 to return the required predicted bounding boxes
    as a ndarray.
    """

    def __init__(self, cfg, gpu_id=None):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
            gpu_id (Optional[int]): GPU id.
        """

        self.cfg = get_cfg()
        self.cfg.merge_from_file(
            model_zoo.get_config_file(cfg.DEMO.DETECTRON2_CFG)
        )
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg.DEMO.DETECTRON2_THRESH
        self.cfg.MODEL.WEIGHTS = cfg.DEMO.DETECTRON2_WEIGHTS
        self.cfg.INPUT.FORMAT = cfg.DEMO.INPUT_FORMAT
        if cfg.NUM_GPUS and gpu_id is None:
            gpu_id = torch.cuda.current_device()
        self.cfg.MODEL.DEVICE = (
            "cuda:{}".format(gpu_id) if cfg.NUM_GPUS > 0 else "cpu"
        )

        logger.info("Initialized Detectron2 Object Detection Model.")

        self.predictor = DefaultPredictor(self.cfg)

    def __call__(self, task):
        """
        Return bounding boxes predictions as a tensor.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames)
        Returns:
            task (TaskInfo object): the same task info object but filled with
                prediction values (a tensor) and the corresponding boxes for
                action detection task.
        """
        middle_frame = task.frames[len(task.frames) // 2]
        outputs = self.predictor(middle_frame)
        # Get only human instances
        mask = outputs["instances"].pred_classes == 0
        pred_boxes = outputs["instances"].pred_boxes.tensor[mask]
        task.add_bboxes(pred_boxes)

        return task
