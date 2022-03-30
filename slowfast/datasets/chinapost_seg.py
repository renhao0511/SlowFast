#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
import torch
import torch.utils.data
from torchvision import transforms

import numpy as np
import json
import cv2
import slowfast.utils.logging as logging
from slowfast.utils.env import pathmgr

from . import decoder as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY
from .random_erasing import RandomErasing
from .transform import create_random_augment

logger = logging.get_logger(__name__)

@DATASET_REGISTRY.register()
class Chinapostseg(torch.utils.data.Dataset):

    def __init__(self, cfg, mode, num_retries=10):
        """
        Construct the ChinaPostSeg video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 path_to_label_1
        path_to_video_2 path_to_label_2
        ...
        path_to_video_N path_to_label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for ChinaPostSeg".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        # if self.mode in ["train", "val"]:
        #     self._num_clips = 1
        # elif self.mode in ["test"]:
        #     self._num_clips = (
        #         cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
        #     )
        self._num_clips = 1

        logger.info("Constructing ChinaPostSeg {}...".format(mode))
        self._construct_loader()
        self.aug = False
        self.rand_erase = False
        self.use_temporal_gradient = False
        self.temporal_gradient_rate = 0.0

        if self.mode == "train" and self.cfg.AUG.ENABLE:
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:
                self.rand_erase = True

    def decode_label_index(self, video_container, path_to_labels, num_frames):
        downsample_rate = 4
        for stride in self.cfg.RESNET.SPATIAL_STRIDES:
            downsample_rate *= stride[0]
        frames_length = 0
        for frame in video_container.decode():
            frames_length += 1
        if frames_length < num_frames:
            return None, None
        video_container.seek(0)
        start_index = random.randint(0, frames_length - num_frames)
        end_index = start_index + num_frames - 1
        frames = []
        for frame in video_container.decode():
            if start_index <= frame.index - frames_length <= end_index:
                frames.append(frame)
        # frames = [frame.to_rgb().to_ndarray() for frame in frames]
        frames_tmp = []
        for frame in frames:
            frame_tmp = frame.to_rgb().to_ndarray()
            h, w = frame_tmp.shape[:2]
            min_hw = min(h, w)
            resize_ratio = self.cfg.DATA.TRAIN_CROP_SIZE / min_hw
            if h < w:
                target_h, target_w = self.cfg.DATA.TRAIN_CROP_SIZE, int((w * resize_ratio + downsample_rate - 1) // downsample_rate * downsample_rate) 
            else:
                target_h, target_w = int((h * resize_ratio + downsample_rate - 1) // downsample_rate * downsample_rate), self.cfg.DATA.TRAIN_CROP_SIZE
            frame_tmp = cv2.resize(frame_tmp, (target_w, target_h))
            frames_tmp.append(frame_tmp)
        frames = frames_tmp
        frames = torch.as_tensor(np.stack(frames))
        frames_shape = frames.shape
        label = torch.zeros((frames_shape[0], frames_shape[1] // downsample_rate, frames_shape[2] // downsample_rate, self.cfg.MODEL.NUM_CLASSES))
        label[:, :, :, 0] = 1
        files = os.listdir(path_to_labels)
        for file in files:
            if not file.endswith('.json'):
                continue
            label_index = int(file.rsplit('.', 1)[0].rsplit('frame', 1)[-1])
            if not start_index <= label_index <= end_index:
                continue
            file_full_path = os.path.join(path_to_labels, file)
            with open(file_full_path, 'r', encoding='utf-8') as fr:
                dict_label = json.load(fr)
            for shape in dict_label['shapes']:
                assert len(shape['points']) == 2, 'points is {}'.format(shape['points'])
                point_lt, point_rb = shape['points']
                label_t, label_b, label_l, label_r = int(point_lt[1] // downsample_rate * resize_ratio), int(point_rb[1] // downsample_rate * resize_ratio), int(point_lt[0] // downsample_rate * resize_ratio), int(point_rb[0] // downsample_rate * resize_ratio)
                label[label_index - start_index, label_t:label_b, label_l:label_r, 1] = 1
                label[label_index - start_index, label_t:label_b, label_l:label_r, 0] = 0
        # for i in range(frames.shape[0]):
        #     index = random.randint(0, 1000)
        #     cv2.imwrite('/home/haoren/repo/slowfast/solution/raw/debug/frames/frame' + str(index) + '.jpg', frames[i].numpy())
        #     cv2.imwrite('/home/haoren/repo/slowfast/solution/raw/debug/label/label' + str(index) + '.jpg', label[i,:,:,1].numpy()*255)
        return frames, label

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
        )
        assert pathmgr.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos = [[]]
        self._path_to_labels = [[]]
        self.inner_index = [0]
        self._spatial_temporal_idx = []
        with pathmgr.open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                if self.cfg.DATA.PATH_LABEL_SEPARATOR not in path_label:
                    self._path_to_videos.append([])
                    self._path_to_labels.append([])
                    self.inner_index.append(0)
                    continue
                assert len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR)) == 2, "path_label is {}, PATH_LABEL_SEPARATOR is {}".format(path_label, self.cfg.DATA.PATH_LABEL_SEPARATOR)
                path, label = path_label.split(
                    self.cfg.DATA.PATH_LABEL_SEPARATOR
                )
                for idx in range(self._num_clips):
                    self._path_to_videos[-1].append(
                        os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                    )
                    self._path_to_labels[-1].append(os.path.join(self.cfg.DATA.PATH_PREFIX, label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load ChinaPostSeg split {} from {}".format(
            self._split_idx, path_to_file
        )
        for i in range(len(self._path_to_videos)):
            assert len(self._path_to_videos[i]) > 0, "Failed to load scenario {}".format(i)
        logger.info(
            "Constructing ChinaPostSeg dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
                + [self.cfg.DATA.TEST_CROP_SIZE]
            )
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )
        # sampling_rate = utils.get_random_sampling_rate(
        #     self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
        #     self.cfg.DATA.SAMPLING_RATE,
        # )
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        flag_full_video = False
        for i_try in range(self._num_retries):
            video_container = None
            try:
                if flag_full_video:
                    video_container = container.get_video_container(
                        self._path_to_videos[index][self.inner_index[-1]],
                        self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                        self.cfg.DATA.DECODING_BACKEND,
                    )
                else:
                    video_container = container.get_video_container(
                        self._path_to_videos[index][self.inner_index[index]],
                        self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                        self.cfg.DATA.DECODING_BACKEND,
                    )
            except Exception as e:
                if flag_full_video:
                    logger.info(
                        "Failed to load video from {} with error {}".format(
                            self._path_to_videos[index][self.inner_index[-1]], e
                        )
                    )
                else:
                    logger.info(
                        "Failed to load video from {} with error {}".format(
                            self._path_to_videos[index][self.inner_index[index]], e
                        )
                    )
            # Select a random video if the current video was not able to access.
            if video_container is None:
                if flag_full_video:
                    logger.warning(
                        "Failed to meta load video idx {} from {}; trial {}".format(
                            index, self._path_to_videos[index][self.inner_index[index]], i_try
                        )
                    )
                else:
                    logger.warning(
                        "Failed to meta load video idx {} from {}; trial {}".format(
                            index, self._path_to_videos[index][self.inner_index[index]], i_try
                        )
                    )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # # Decode video. Meta info is used to perform selective decoding.
            # frames = decoder.decode(
            #     video_container,
            #     sampling_rate,
            #     self.cfg.DATA.NUM_FRAMES,
            #     temporal_sample_index,
            #     self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
            #     video_meta=self._video_meta[index],
            #     target_fps=self.cfg.DATA.TARGET_FPS,
            #     backend=self.cfg.DATA.DECODING_BACKEND,
            #     max_spatial_scale=min_scale,
            #     use_offset=self.cfg.DATA.USE_OFFSET_SAMPLING,
            # )

            if flag_full_video:
                frames, label = self.decode_label_index(video_container, self._path_to_labels[index][self.inner_index[-1]], self.cfg.DATA.NUM_FRAMES)
            else:
                frames, label = self.decode_label_index(video_container, self._path_to_labels[index][self.inner_index[index]], self.cfg.DATA.NUM_FRAMES)

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                if flag_full_video:
                    logger.warning(
                        "Failed to decode video idx {} from {}; trial {}".format(
                            index, self._path_to_videos[index][self.inner_index[-1]], i_try
                        )
                    )
                else:
                    logger.warning(
                        "Failed to decode video idx {} from {}; trial {}".format(
                            index, self._path_to_videos[index][self.inner_index[index]], i_try
                        )
                    )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            self.inner_index[index] += 1
            if self.inner_index[index] >= len(self._path_to_videos[index]):
                self.inner_index[index] -= len(self._path_to_videos[index])
            flag_full_video = not flag_full_video

            if self.aug:
                if self.cfg.AUG.NUM_SAMPLE > 1:

                    frame_list = []
                    label_list = []
                    index_list = []
                    for _ in range(self.cfg.AUG.NUM_SAMPLE):
                        new_frames = self._aug_frame(
                            frames,
                            spatial_sample_index,
                            min_scale,
                            max_scale,
                            crop_size,
                        )
                        # label = self._path_to_labels[index]
                        new_frames = utils.pack_pathway_output(
                            self.cfg, new_frames
                        )
                        frame_list.append(new_frames)
                        label_list.append(label)
                        index_list.append(index)
                    return frame_list, label_list, index_list, {}

                else:
                    frames = self._aug_frame(
                        frames,
                        spatial_sample_index,
                        min_scale,
                        max_scale,
                        crop_size,
                    )

            else:
                frames = utils.tensor_normalize(
                    frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
                )
                # T H W C -> C T H W.
                frames = frames.permute(3, 0, 1, 2)
                # # Perform data augmentation.
                # frames = utils.spatial_sampling(
                #     frames,
                #     spatial_idx=spatial_sample_index,
                #     min_scale=min_scale,
                #     max_scale=max_scale,
                #     crop_size=crop_size,
                #     random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                #     inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                # )

            # label = self._path_to_labels[index]
            frames = utils.pack_pathway_output(self.cfg, frames)
            return frames, label, index, {}
        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )

    def _aug_frame(
        self,
        frames,
        spatial_sample_index,
        min_scale,
        max_scale,
        crop_size,
    ):
        aug_transform = create_random_augment(
            input_size=(frames.size(1), frames.size(2)),
            auto_augment=self.cfg.AUG.AA_TYPE,
            interpolation=self.cfg.AUG.INTERPOLATION,
        )
        # T H W C -> T C H W.
        frames = frames.permute(0, 3, 1, 2)
        list_img = self._frame_to_list_img(frames)
        list_img = aug_transform(list_img)
        frames = self._list_img_to_frames(list_img)
        frames = frames.permute(0, 2, 3, 1)

        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE,
            self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE,
        )
        relative_scales = (
            None if (self.mode not in ["train"] or len(scl) == 0) else scl
        )
        relative_aspect = (
            None if (self.mode not in ["train"] or len(asp) == 0) else asp
        )
        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            aspect_ratio=relative_aspect,
            scale=relative_scales,
            motion_shift=self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT
            if self.mode in ["train"]
            else False,
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                self.cfg.AUG.RE_PROB,
                mode=self.cfg.AUG.RE_MODE,
                max_count=self.cfg.AUG.RE_COUNT,
                num_splits=self.cfg.AUG.RE_COUNT,
                device="cpu",
            )
            frames = frames.permute(1, 0, 2, 3)
            frames = erase_transform(frames)
            frames = frames.permute(1, 0, 2, 3)

        return frames

    def _frame_to_list_img(self, frames):
        img_list = [
            transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))
        ]
        return img_list

    def _list_img_to_frames(self, img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
