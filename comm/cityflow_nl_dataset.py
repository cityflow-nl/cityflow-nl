"""
PyTorch dataset for CityFlow-NL.
"""
import json
import os
import random

import cv2
import torch
from detectron2.structures import BoxMode
from torch.utils.data import Dataset
from tqdm import tqdm
from yacs.config import CfgNode as CN

from comm.utils import TqdmToLogger, get_logger


DATA_CFG = CN()
DATA_CFG.HOME = "/home/"
DATA_CFG.CITYFLOW_PATH = "data/cityflow"
DATA_CFG.JSON_PATH = "data/train-tracks.json"
DATA_CFG.CROP_SIZE = (128, 128)
DATA_CFG.POSITIVE_THRESHOLD = 0.5
DATA_CFG.EVAL_TRACKS_JSON_PATH = "data/test-tracks.json"
DATA_CFG.FORD2 = False


def get_default_data_config():
    return DATA_CFG.clone()


class CityFlowNLDataset(Dataset):
    def __init__(self, data_cfg, num_tracks=-1):
        """
        Dataset for training.
        :param data_cfg: CfgNode for CityFlow NL.
        """
        self.data_cfg = data_cfg.clone()
        with open(self.data_cfg.JSON_PATH) as f:
            tracks = json.load(f)
        self.list_of_uuids = list(tracks.keys())
        self.list_of_tracks = list(tracks.values())
        self.list_of_crops = list()

        gt_boxes_dict = dict()
        if num_tracks == -1:
            num_tracks = len(self.list_of_tracks)
        pbar = tqdm(total=num_tracks, leave=False, desc="Loading Data", file=TqdmToLogger(),
                    mininterval=1, maxinterval=100, )
        for track in self.list_of_tracks[:num_tracks]:
            scene = "/".join(track["frames"][0].split("/")[0:3])
            if scene in gt_boxes_dict:
                gt_boxes = gt_boxes_dict[scene]
            else:
                gt_file = os.path.join(
                    self.data_cfg.HOME, self.data_cfg.CITYFLOW_PATH, scene, "gt", "gt.txt")
                with open(gt_file) as f:
                    gt_boxes = f.readlines()
                gt_boxes_dict[scene] = gt_boxes
            for frame_idx, frame in enumerate(track["frames"]):
                frame_path = os.path.join(
                    self.data_cfg.HOME, self.data_cfg.CITYFLOW_PATH, frame)
                nl_idx = int(random.uniform(0, 3))
                nl = track["nl"][nl_idx]
                box = track["boxes"][frame_idx]
                other_boxes = self._get_all_other_objects_in_frame(
                    frame.split("/")[4], box, gt_boxes)
                crop = {"frame": frame_path,
                        "nl": nl,
                        "box": box,
                        "other_boxes": other_boxes}
                if self.data_cfg.FORD2:
                    self._prepare_for_d2(crop)
                del crop["other_boxes"]
                self.list_of_crops.append(crop)
            pbar.update()
        pbar.close()
        self._logger = get_logger()

    def __len__(self):
        return len(self.list_of_crops)

    def __getitem__(self, index):
        """
        Get pairs of NL and cropped frame.
        """
        dp = self.list_of_crops[index]
        if not os.path.isfile(dp["frame"]):
            self._logger.warning("Missing Image File: %s" % dp["frame"])
            raise FileNotFoundError(dp["frame"])

        if random.uniform(0, 1) > self.data_cfg.POSITIVE_THRESHOLD:
            label = 1
        else:
            label = 0
        if label != 1:
            dp["nl"] = random.sample(self.list_of_crops, 1)[0]["nl"]
        dp["label"] = torch.Tensor([label]).to(dtype=torch.float32)
        if self.data_cfg.FORD2:
            return dp

        frame = cv2.imread(dp["frame"])
        # dp["bgr_img"] = torch.from_numpy(frame).permute(
        #     [2, 0, 1]).to(dtype=torch.float32)
        box = dp["box"]
        crop = frame[box[1]:box[1] + box[3], box[0]: box[0] + box[2], :]
        crop = cv2.resize(crop, dsize=self.data_cfg.CROP_SIZE)
        crop = torch.from_numpy(crop).permute([2, 0, 1]).to(
            dtype=torch.float32)
        dp["crop"] = crop
        return dp

    @staticmethod
    def _prepare_for_d2(dp):
        dp["file_name"] = dp["frame"]
        dp["annotations"] = list()
        target_box = {"bbox": dp["box"],
                      "bbox_mode": BoxMode.XYWH_ABS, "category_id": 0}
        dp["annotations"].append(target_box)
        for box in dp["other_boxes"]:
            other_box = {"bbox": box,
                         "bbox_mode": BoxMode.XYWH_ABS, "category_id": 1}
            dp["annotations"].append(other_box)

    @staticmethod
    def _get_all_other_objects_in_frame(frame, current_box, gt_boxes):
        frame_id = int(frame.split(".")[0])
        objs = [row for row in gt_boxes if row.startswith("%d," % frame_id)]
        other_boxes = []
        for obj in objs:
            box = [int(axis) for axis in obj.split(",")[2:6]]
            if box != current_box:
                other_boxes.append(box)
        return other_boxes


class CityFlowNLInferenceDataset(Dataset):
    def __init__(self, data_cfg):
        """Dataset for evaluation. Loading tracks instead of frames."""
        self.data_cfg = data_cfg
        with open(self.data_cfg.EVAL_TRACKS_JSON_PATH) as f:
            tracks = json.load(f)
        self.list_of_uuids = list(tracks.keys())
        self.list_of_tracks = list(tracks.values())
        self._logger = get_logger()

    def __len__(self):
        return len(self.list_of_uuids)

    def __getitem__(self, index):
        """
        :return: a dictionary for each track:
        id: uuid for the track
        frames, boxes, nl are untouched from the input json file.
        crops: A Tensor of cropped images from the track of shape
            [length, 3, crop_w, crop_h].
        """
        dp = {"id": self.list_of_uuids[index]}
        dp.update(self.list_of_tracks[index])
        cropped_frames = []
        for frame_idx, frame_path in enumerate(dp["frames"]):
            frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, frame_path)
            if not os.path.isfile(frame_path):
                continue
            frame = cv2.imread(frame_path)
            box = dp["boxes"][frame_idx]
            crop = frame[box[1]:box[1] + box[3], box[0]: box[0] + box[2], :]
            crop = cv2.resize(crop, dsize=self.data_cfg.CROP_SIZE)
            crop = torch.from_numpy(crop).permute([2, 0, 1]).to(
                dtype=torch.float32)
            cropped_frames.append(crop)
        dp["crops"] = torch.stack(cropped_frames, dim=0)
        return dp


class CityFlowNLTrackingInferenceDataset(Dataset):
    def __init__(self, data_cfg, camera, tracker):
        """Dataset for evaluation. Loading tracks instead of frames."""
        self.data_cfg = data_cfg

        tracks = _get_mtsc_tracks(os.path.join(
            self.data_cfg.HOME, self.data_cfg.CITYFLOW_PATH), camera, tracker)
        self.list_of_uuids = list(tracks.keys())
        self.list_of_tracks = list(tracks.values())
        self._logger = get_logger()

    def __len__(self):
        return len(self.list_of_uuids)

    def __getitem__(self, index):
        """
        :return: a dictionary for each track:
        id: uuid for the track
        frames, boxes, nl are untouched from the input json file.
        crops: A Tensor of cropped images from the track of shape
            [length, 3, crop_w, crop_h].
        """
        dp = {"id": self.list_of_uuids[index]}
        dp.update(self.list_of_tracks[index])
        cropped_frames = []
        selected_frames = list(zip(dp["frames"], dp["boxes"]))
        if len(selected_frames) > 16:
            selected_frames = random.sample(selected_frames, 16)
        for f in selected_frames:
            frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, f[0])
            if not os.path.isfile(frame_path):
                continue
            frame = cv2.imread(frame_path)
            box = f[1]
            try:
                box[2] = min(box[2], frame.shape[1]-box[0])
                box[3] = min(box[3], frame.shape[0]-box[1])
                crop = frame[box[1]:box[1] + box[3],
                             box[0]: box[0] + box[2], :]
                crop = cv2.resize(crop, dsize=self.data_cfg.CROP_SIZE)
                crop = torch.from_numpy(crop).permute([2, 0, 1]).to(
                    dtype=torch.float32)
            except Exception:
                continue
            cropped_frames.append(crop)
        dp["crops"] = torch.stack(cropped_frames, dim=0)
        return dp


def _get_mtsc_tracks(path_to_cityflow, cam, tracker):
    tracks = dict()
    with open(os.path.join(path_to_cityflow, cam.strip(), "mtsc", tracker+".txt")) as f:
        lines = f.readlines()
    for line in lines:
        line = line.split(",")
        if line[1] in tracks:
            pass
        else:
            tracks[line[1]] = {"frames": [], "boxes": []}

        box = [int(float(line[2])), int(float(line[3])),
               int(float(line[4])), int(float(line[5]))]
        box[0] = max(box[0], 0)
        box[1] = max(box[1], 0)
        if box[2] < 5 or box[3] < 5:
            continue
        tracks[line[1]]["frames"].append(os.path.join(
            cam, "img1", "%06d.jpg") % int(line[0]))
        tracks[line[1]]["boxes"].append(box)
    return tracks


class CityFlowNLVTNDataset(Dataset):
    def __init__(self, data_cfg, frames):
        """Dataset for evaluation. Loading tracks instead of frames."""
        self.data_cfg = data_cfg
        self.frames = frames
        self._logger = get_logger()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        """
        :return: a dictionary for each track:
        id: uuid for the track
        frames, boxes, nl are untouched from the input json file.
        crops: A Tensor of cropped images from the track of shape
            [length, 3, crop_w, crop_h].
        """
        frame_path = os.path.join(
            self.data_cfg.HOME, self.data_cfg.CITYFLOW_PATH, self.frames[index])
        dp = {"file_name": frame_path}
        return dp
