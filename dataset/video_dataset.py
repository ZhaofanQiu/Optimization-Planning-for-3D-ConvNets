"""
Dataset to load video/frames/LMDB
By Zhaofan Qiu
zhaofanqiu@gmail.com
"""
import torch.utils.data
import os
import random
import torch
import numpy as np
import lmdb
import io
from PIL import Image
import cv2


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, list_file, transform, root_path, clip_length=1, num_steps=1, num_segments=1, num_channels=3,
                 format="LMDB"):
        super(VideoDataset, self).__init__()
        self.list_file = list_file
        self.transform = transform
        self.root_path = root_path
        self.clip_length = clip_length
        self.num_steps = num_steps
        self.num_segments = num_segments
        self.num_channels = num_channels
        self.format = format

        self.samples = self._load_list(list_file)

    def _load_list(self, list_root):
        with open(list_root, 'r') as f:
            samples = f.readlines()
        return samples

    def _parse_rgb_lmdb(self, video_path, offsets):
        """Return the clip buffer sample from video lmdb."""
        lmdb_env = lmdb.open(os.path.join(self.root_path, video_path), readonly=True)

        with lmdb_env.begin() as lmdb_txn:
            image_list = []
            for offset in offsets:
                for frame_id in range(offset + 1, offset + self.num_steps * self.clip_length + 1, self.num_steps):
                    bio = io.BytesIO(lmdb_txn.get('{:06d}.jpg'.format(frame_id).encode()))
                    image = Image.open(bio).convert('RGB')
                    image_list.append(image)
        lmdb_env.close()
        return image_list

    def _parse_rgb_raw(self, video_path, offsets):
        """Return the clip buffer sample from raw video."""
        cap = cv2.VideoCapture(os.path.join(self.root_path, video_path))

        image_list = []
        pos_list = []
        current_pos = 0
        for offset in offsets:
            for frame_id in range(offset + 1, offset + self.num_steps * self.clip_length + 1, self.num_steps):
                if (frame_id - 1) in pos_list:
                    idx = pos_list.index(frame_id - 1)
                    image_list.append(image_list[idx].copy())
                else:
                    if not frame_id - 1 == current_pos:
                        if (frame_id - 1 - current_pos > 0) and (frame_id - 1 - current_pos <= 16):
                            while frame_id - 1 - current_pos > 0:
                                success, frame = cap.read()
                                current_pos += 1
                        else:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
                            current_pos = frame_id - 1

                    # assert frame_id - 1 == current_pos
                    success, frame = cap.read()
                    current_pos += 1
                    if not success:
                        print(video_path)
                        print(frame_id - 1)
                        assert success

                    image = Image.fromarray(frame[..., ::-1])
                    image_list.append(image)
                pos_list.append(frame_id - 1)
        cap.release()
        return image_list

    def _parse_rgb_frame(self, video_path, offsets):
        """Return the clip buffer sample from video frames."""
        image_list = []
        for offset in offsets:
            for frame_id in range(offset + 1, offset + self.num_steps * self.clip_length + 1, self.num_steps):
                image = Image.open(os.path.join(self.root_path, video_path, '{:05d}'.format(frame_id) + '.jpg')).convert('RGB')
                image_list.append(image)
        return image_list

    def _parse_flow_lmdb(self, video_path, offsets):
        """Return the clip buffer sample from video lmdb."""
        lmdb_env = lmdb.open(os.path.join(self.root_path, video_path), readonly=True)

        with lmdb_env.begin() as lmdb_txn:
            image_list = []
            for offset in offsets:
                for frame_id in range(offset + 1, offset + self.num_steps * self.clip_length + 1, self.num_steps):
                    bio = io.BytesIO(lmdb_txn.get('{:06d}_c1.jpg'.format(frame_id).encode()))
                    image = Image.open(bio)
                    image_list.append(image)
                    bio = io.BytesIO(lmdb_txn.get('{:06d}_c2.jpg'.format(frame_id).encode()))
                    image = Image.open(bio)
                    image_list.append(image)
        lmdb_env.close()
        return image_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        raise NotImplementedError


class VideoTrainDataset(VideoDataset):
    def _parse_sample_str(self, sample, video_idx):
        ss = sample.split(' ')
        video_path = sample[:-len(ss[-1]) - 1 - len(ss[-2]) - 1]
        duration = int(ss[-2])

        label = int(ss[-1][:-1])

        # sample frames offsets
        offsets = []
        length_ext = self.clip_length * self.num_steps
        ave_duration = duration // self.num_segments
        if ave_duration >= length_ext:
            for i in range(self.num_segments):
                offsets.append(random.randint(0, ave_duration - length_ext) + i * ave_duration)
        else:
            if duration >= length_ext:
                float_ave_duration = float(duration - length_ext) / float(self.num_segments)
                for i in range(self.num_segments):
                    offsets.append(random.randint(0, int(float_ave_duration)) + int(i * float_ave_duration))
            else:
                raise NotImplementedError
        return video_path, offsets, label


class VideoTestDataset(VideoDataset):
    def __init__(self, list_file, num_clips, transform, root_path, clip_length=1, num_steps=1, num_segments=1,
                 num_channels=3, format="LMDB"):
        super(VideoTestDataset, self).__init__(list_file, transform, root_path, clip_length, num_steps, num_segments,
                                               num_channels, format)
        self.num_clips = num_clips

    def __len__(self):
        return len(self.samples) * self.num_clips

    def _parse_sample_str(self, sample, video_idx, clip_idx):
        ss = sample.split(' ')
        video_path = sample[:-len(ss[-1]) - 1 - len(ss[-2]) - 1]
        duration = int(ss[-2])

        label = int(ss[-1][:-1])

        # sample frames offsets
        offsets = []
        length_ext = self.clip_length * self.num_steps
        ave_duration = duration // self.num_segments
        if ave_duration >= length_ext:
            for i in range(self.num_segments):
                offsets.append(int(float(ave_duration - length_ext) * clip_idx / self.num_clips) + i * ave_duration)
        else:
            if duration >= length_ext:
                float_ave_duration = float(duration - length_ext) / float(self.num_segments)
                for i in range(self.num_segments):
                    offsets.append(
                        int(float_ave_duration * clip_idx / self.num_clips) + int(i * float_ave_duration))
            else:
                raise NotImplementedError
        return video_path, offsets, label


class VideoRGBTrainDataset(VideoTrainDataset):
    def __getitem__(self, item):
        video_path, offsets, label = self._parse_sample_str(self.samples[item], item)
        if self.format == "LMDB":
            image_list = self._parse_rgb_lmdb(video_path, offsets)
        elif self.format == "RAW":
            image_list = self._parse_rgb_raw(video_path, offsets)
        elif self.format == "FRAME":
            image_list = self._parse_rgb_frame(video_path, offsets)
        else:
            raise NotImplementedError
        trans_image_list = self.transform(image_list)
        return trans_image_list, label, item


class VideoRGBTestDataset(VideoTestDataset):
    def __getitem__(self, item):
        item_in = item % self.num_clips
        item_out = item // self.num_clips

        video_path, offsets, label = self._parse_sample_str(self.samples[item_out], item_out, item_in)
        if self.format == "LMDB":
            image_list = self._parse_rgb_lmdb(video_path, offsets)
        elif self.format == "RAW":
            image_list = self._parse_rgb_raw(video_path, offsets)
        elif self.format == "FRAME":
            image_list = self._parse_rgb_frame(video_path, offsets)
        else:
            raise NotImplementedError
        trans_image_list = self.transform(image_list)
        return trans_image_list, label


class VideoFlowTrainDataset(VideoTrainDataset):
    def __getitem__(self, item):
        video_path, offsets, label = self._parse_sample_str(self.samples[item], item)
        if self.format == "LMDB":
            image_list = self._parse_flow_lmdb(video_path, offsets)
        else:
            raise NotImplementedError
        trans_image_list = self.transform(image_list)
        return trans_image_list, label, item


class VideoFlowTestDataset(VideoTestDataset):
    def __getitem__(self, item):
        item_in = item % self.num_clips
        item_out = item // self.num_clips

        video_path, offsets, label = self._parse_sample_str(self.samples[item_out], item_out, item_in)
        if self.format == "LMDB":
            image_list = self._parse_flow_lmdb(video_path, offsets)
        else:
            raise NotImplementedError
        trans_image_list = self.transform(image_list)
        return trans_image_list, label