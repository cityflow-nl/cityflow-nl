"""
Default data i/o configurations and PyTorch dataset for CityFlow-NL.
"""
import gzip
import json
import os
import pickle
import random
import uuid

import PIL
import torch
import torchvision
from fvcore.common.config import CfgNode as CN
from torch.utils.data import Dataset
from tqdm import tqdm

from comm.utils import TqdmToLogger, get_logger

DATA_CFG = CN()
DATA_CFG.HOME = "/home/"
DATA_CFG.CITYFLOW_PATH = "data/cityflow"
DATA_CFG.JSON_PATH = "data/train-tracks_nlpaug_with_target_id.json"
DATA_CFG.CROP_SIZE = (256, 256)
DATA_CFG.USE_OPTICAL_FLOW = True
DATA_CFG.POSITIVE_THRESHOLD = 0.0
DATA_CFG.FORD2 = False
DATA_CFG.FRAME_PER_TRACK = 32
DATA_CFG.SPLIT = "train"


def get_default_data_config():
    """Get default data configurations.

    Returns:
        CfgNode: A copy of the default data config.
    """
    return DATA_CFG.clone()


class CityFlowNLDataset(Dataset):
    """Dataset for training."""

    def __init__(self, data_cfg, num_tracks=-1):
        """
        :param data_cfg: CfgNode for CityFlow NL.
        """
        super().__init__()
        self.data_cfg = data_cfg.clone()
        self._logger = get_logger("%s_dataset" % self.data_cfg.SPLIT)
        with open(self.data_cfg.JSON_PATH) as track_file:
            tracks = json.load(track_file)
        self.list_of_sampled_tracks = self._prepare_dataset(tracks, num_tracks)
        self.resize_op = torchvision.transforms.Resize(
            size=self.data_cfg.CROP_SIZE)
        self.pil_to_tensor = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.list_of_sampled_tracks)

    def __getitem__(self, index):
        """
        Get pairs of NL and cropped frame.
        """
        track = self.list_of_sampled_tracks[index].copy()
        crop = dict()
        frame = dict(random.sample(track["crops"], 1)[0])
        if not os.path.isfile(frame["frame_path"]):
            self._logger.warning(
                "Missing Image File: %s", frame["frame_path"])
            raise FileNotFoundError(frame["frame_path"])

        crop["target_id"] = track["target_id"]
        img = PIL.Image.open(frame["frame_path"])
        box = frame["box"]
        img_crop = img.crop((box[0], box[1], box[0] + box[2], box[1]+box[3]))
        img_crop = self.pil_to_tensor(img_crop)
        img_crop = self.resize_op(img_crop)
        crop["crop"] = img_crop
        img = self.pil_to_tensor(img)
        img = self.resize_op(img)
        crop["full_frame"] = img
        if self.data_cfg.USE_OPTICAL_FLOW:
            crop["flows"] = torch.load(track["stacked_flow_path"])
        crop["random_nl"] = random.sample(track["nl"], 1)[0]
        return crop

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

    def _prepare_dataset(self, tracks, num_tracks):
        gt_boxes_dict = dict()
        if num_tracks == -1:
            num_tracks = len(tracks)
        pbar = tqdm(total=num_tracks, leave=False, desc="Preparing Data", file=TqdmToLogger(),
                    mininterval=1, maxinterval=100, )
        list_of_sampled_tracks = []
        for track_id in list(tracks.keys())[:num_tracks]:
            list_of_crops = []
            track = tracks[track_id]
            scene = "/".join(track["frames"][0].split("/")[0:3])
            if scene in gt_boxes_dict:
                gt_boxes = gt_boxes_dict[scene]
            else:
                with open(os.path.join(self.data_cfg.HOME,
                                       self.data_cfg.CITYFLOW_PATH,
                                       scene,
                                       "gt",
                                       "gt.txt")) as gt_file:
                    gt_boxes = gt_file.readlines()
                gt_boxes_dict[scene] = gt_boxes
            for frame_idx, frame in enumerate(track["frames"]):
                frame_path = os.path.join(
                    self.data_cfg.HOME, self.data_cfg.CITYFLOW_PATH, frame)
                box = track["boxes"][frame_idx]
                other_boxes = self._get_all_other_objects_in_frame(
                    frame.split("/")[4], box, gt_boxes)
                if self.data_cfg.USE_OPTICAL_FLOW:
                    path_to_flow = os.path.join(
                        self.data_cfg.HOME, self.data_cfg.CITYFLOW_PATH,
                        "flow",
                        "/".join(frame.split("/")[:3]),
                        "flow",
                        "%s.pklz" % frame.split("/")[-1].split(".")[0]
                    )
                else:
                    path_to_flow = ""
                crop = {"frame_path": frame_path,
                        "box": box,
                        "other_boxes": other_boxes,
                        "flow_path": path_to_flow,
                        }
                list_of_crops.append(crop)
            stacked_flow_path = os.path.join(
                self.data_cfg.HOME, self.data_cfg.CITYFLOW_PATH,
                "stacked_crop_flows",
                "%s.pth" % track_id
            )
            stacked_frames_path = os.path.join(
                self.data_cfg.HOME, self.data_cfg.CITYFLOW_PATH,
                "track_stacked_imgs",
                "%s.pklz" % track_id
            )
            if self.data_cfg.SPLIT == "train":
                if len(list_of_crops) > 1:
                    list_of_sampled_tracks.append(
                        {"track_id": uuid.UUID(track_id).bytes,
                         "crops": list_of_crops,
                         "nl": track["nl"],
                         "target_id": track["target_id"],
                         "stacked_flow_path": stacked_flow_path,
                         "stacked_frames_path": stacked_frames_path})
                else:
                    self._logger.warning(
                        "Track %s only have 1 frame.", track_id)
            else:
                list_of_sampled_tracks.append(
                    {"track_id": uuid.UUID(track_id).bytes,
                     "crops": list_of_crops,
                     "stacked_flow_path": stacked_flow_path,
                     "stacked_frames_path": stacked_frames_path})
            pbar.update()
        pbar.close()
        if self.data_cfg.SPLIT == "train":
            list_of_sampled_tracks = list_of_sampled_tracks * 2000
        return list_of_sampled_tracks


class CityFlowNLInferenceDataset(CityFlowNLDataset):
    """Dataset for evaluation. Loading tracks instead of frames."""

    def __len__(self):
        return len(self.list_of_sampled_tracks)

    def __getitem__(self, index):
        """
        Get pairs of NL and cropped frame.
        """
        track = self.list_of_sampled_tracks[index]
        if not ("crop" in track and "full_frame" in track and "flows" in track):
            track["crop"] = []
            track["full_frame"] = []
            list_of_frames = track["crops"]
            num_samples = min(self.data_cfg.FRAME_PER_TRACK,
                              len(list_of_frames))
            list_of_frames = random.sample(
                list_of_frames, num_samples)
            with gzip.open(track["stacked_frames_path"], 'rb') as flow_file:
                stacked_frames = pickle.load(flow_file)
            boxes = [f["box"] for f in list_of_frames]
            for frame, box in zip(stacked_frames, boxes):
                frame = PIL.Image.fromarray(frame)
                crop = frame.crop(
                    (box[0], box[1], box[0] + box[2], box[1]+box[3]))
                crop = self.pil_to_tensor(crop)
                crop = self.resize_op(crop)
                track["crop"].append(crop)
                frame = self.pil_to_tensor(frame)
                frame = self.resize_op(frame)
                track["full_frame"].append(frame)
            track["crop"] = torch.stack(track["crop"])
            track["full_frame"] = torch.stack(track["full_frame"])
            if self.data_cfg.USE_OPTICAL_FLOW:
                track["flows"] = torch.load(track["stacked_flow_path"])
        return track


class CityFlowNLInferenceSimilarityLoadingDataset(Dataset):
    """Dataset for training."""

    def __init__(self, path_to_embeddings):
        super().__init__()
        self.path_to_embeddings = os.path.join(
            path_to_embeddings, "visual_embeds")
        self.path_to_visual_embeds = [f for f in os.listdir(
            self.path_to_embeddings) if os.path.isfile(os.path.join(self.path_to_embeddings, f))]

    def __len__(self):
        return len(self.path_to_visual_embeds)

    def __getitem__(self, index):
        path = self.path_to_visual_embeds[index]
        embeds = torch.load(os.path.join(self.path_to_embeddings, path))
        embeds["track_id"] = uuid.UUID(path.split(".")[0]).bytes
        return embeds
