"""Vehicle Retrieval Network MK-II."""

import os
import time
import uuid

import torch
import torchvision

from comm.language_models import CLIPEmbeddingModel
from third_party.senet import se_resnext50_32x4d


class VehicleRetrievalNetwork(torch.nn.Module):
    """Vehicle Retrieval Network MK-II"""

    def __init__(self, model_cfg) -> None:
        super().__init__()
        self.model_cfg = model_cfg.clone()

        # Language Models.
        self.clip_model = CLIPEmbeddingModel(self.model_cfg)
        self.embed_dim = self.clip_model.nl_embed_dim
        self.lang_fc = torch.nn.Sequential(
            torch.nn.LayerNorm(self.embed_dim),
            torch.nn.Linear(self.embed_dim, self.embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embed_dim, self.embed_dim))

        # Vision Models.
        self.seresnet_crop = se_resnext50_32x4d()
        self.seresnet_crop_fc = torch.nn.Conv2d(
            2048, self.embed_dim, kernel_size=1)
        self.clip_crop_fc = torch.nn.Sequential(
            torch.nn.Linear(self.clip_model.vision_embed_dim, self.embed_dim),
            torch.nn.BatchNorm1d(self.embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embed_dim, self.embed_dim))
        self.clip_frame_fc = torch.nn.Sequential(
            torch.nn.Linear(self.clip_model.vision_embed_dim, self.embed_dim),
            torch.nn.BatchNorm1d(self.embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embed_dim, self.embed_dim))
        self.flow_conv = torch.nn.Sequential(
            torch.nn.Conv2d(2, 32, kernel_size=3, padding=1),
            torch.nn.MaxPool2d(3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 128, kernel_size=3, padding=1),
            torch.nn.MaxPool2d(3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(8 * 8 * 128, self.embed_dim))
        self.flow_gru = torch.nn.GRU(
            self.embed_dim, self.embed_dim, batch_first=True)

        self.merge_fc = torch.nn.Sequential(
            torch.nn.Linear(4 * self.embed_dim, self.embed_dim),
            torch.nn.BatchNorm1d(self.embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embed_dim, self.embed_dim))

        self.temperature = {
            "crop_embeds": torch.nn.Parameter(torch.ones(()), requires_grad=True),
            "flow_embeds": torch.nn.Parameter(torch.ones(()), requires_grad=True),
            "clip_frame_embeds": torch.nn.Parameter(torch.ones(()), requires_grad=True),
            "clip_crop_embeds": torch.nn.Parameter(torch.ones(()), requires_grad=True),
            "merge_embeds": torch.nn.Parameter(torch.ones(()), requires_grad=True),
        }
        self.sim_weights = {
            "crop_embeds_sims": torch.nn.Parameter(torch.ones(()), requires_grad=True),
            "flow_embeds_sims": torch.nn.Parameter(torch.ones(()), requires_grad=True),
            "clip_frame_embeds_sims": torch.nn.Parameter(torch.ones(()), requires_grad=True),
            "clip_crop_embeds_sims": torch.nn.Parameter(torch.ones(()), requires_grad=True),
            "merge_embeds_sims": torch.nn.Parameter(torch.ones(()), requires_grad=True),
        }

        self.lang_crop_fc = torch.nn.Sequential(
            torch.nn.LayerNorm(self.embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embed_dim, self.embed_dim))
        self.lang_clip_crop_fc = torch.nn.Sequential(
            torch.nn.LayerNorm(self.embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embed_dim, self.embed_dim))
        self.lang_clip_frame_fc = torch.nn.Sequential(
            torch.nn.LayerNorm(self.embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embed_dim, self.embed_dim))
        self.lang_flow_fc = torch.nn.Sequential(
            torch.nn.LayerNorm(self.embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embed_dim, self.embed_dim))
        self.lang_merge_fc = torch.nn.Sequential(
            torch.nn.LayerNorm(self.embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embed_dim, self.embed_dim))

        self.crop_id_loss = torch.nn.Sequential(
            torch.nn.Linear(self.embed_dim, self.embed_dim),
            torch.nn.BatchNorm1d(self.embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embed_dim, self.model_cfg.NUM_CLASS))
        self.clip_crop_id_loss = torch.nn.Sequential(
            torch.nn.Linear(self.embed_dim, self.embed_dim),
            torch.nn.BatchNorm1d(self.embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embed_dim, self.model_cfg.NUM_CLASS))
        self.clip_frame_id_loss = torch.nn.Sequential(
            torch.nn.Linear(self.embed_dim, self.embed_dim),
            torch.nn.BatchNorm1d(self.embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embed_dim, self.model_cfg.NUM_CLASS))
        self.merge_id_loss = torch.nn.Sequential(
            torch.nn.Linear(self.embed_dim, self.embed_dim),
            torch.nn.BatchNorm1d(self.embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embed_dim, self.model_cfg.NUM_CLASS))

        self.rand_aug = torchvision.transforms.Compose([
            torchvision.transforms.RandomApply(
                [torchvision.transforms.RandomRotation(10)], p=0.5),
            torchvision.transforms.RandomApply(
                [torchvision.transforms.ColorJitter(brightness=.5, hue=.3)], p=0.5),
        ])
        self.imagenet_normalization = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.to_pil = torchvision.transforms.ToPILImage()

    def compute_nl_embeds(self, nls):
        """Compute the NL embedding from the language model.

        Args:
            nls (List[str]): A list of "num_nls" of NL queries.

        Returns:
            Tensor: num_nls * self.embed_dim
        """
        nl_embeds = self.clip_model(nls)
        lang_embeds = self.lang_fc(nl_embeds)
        lang_crop_embeds = self.lang_crop_fc(lang_embeds)
        lang_flow_embeds = self.lang_flow_fc(lang_embeds)
        lang_clip_crop_embeds = self.lang_clip_crop_fc(lang_embeds)
        lang_clip_frame_embeds = self.lang_clip_frame_fc(lang_embeds)
        lang_merge_embeds = self.lang_merge_fc(lang_embeds)
        return {
            "lang_crop_embeds": lang_crop_embeds,
            "lang_flow_embeds": lang_flow_embeds,
            "lang_clip_crop_embeds": lang_clip_crop_embeds,
            "lang_clip_frame_embeds": lang_clip_frame_embeds,
            "lang_merge_embeds": lang_merge_embeds,
        }

    def _compute_crop_embeds(self, crops):
        """Compute the crop embedding from the se-resnet model."""
        crops = crops.cuda()
        if self.training:
            crops = self.rand_aug(crops)
        crops = self.imagenet_normalization(crops)
        crop_embeds = self.seresnet_crop_fc(self.seresnet_crop(crops))
        crop_embeds = crop_embeds.view(crop_embeds.size(0), -1)
        cls_logits = self.crop_id_loss(crop_embeds) if self.training else None
        return crop_embeds, cls_logits

    def _compute_clip_crop_embeds(self, crops):
        "Compute the crop embedding from the CLIP model."
        if self.training:
            crops = self.rand_aug(crops)
        clip_crop_embeds = self.clip_crop_fc(self.clip_model.compute_visual_embedding(
            [self.to_pil(crop) for crop in crops]))
        cls_logits = self.clip_crop_id_loss(
            clip_crop_embeds) if self.training else None
        return clip_crop_embeds, cls_logits

    def _compute_clip_frame_embeds(self, full_frames):
        "Compute the frame embedding from the CLIP model."
        if self.training:
            full_frames = self.rand_aug(full_frames)
        clip_frame_embeds = self.clip_frame_fc(self.clip_model.compute_visual_embedding(
            [self.to_pil(frame) for frame in full_frames]))
        cls_logits = self.clip_frame_id_loss(clip_frame_embeds)
        return clip_frame_embeds, cls_logits

    def _compute_flow_embeds(self, flows):
        "Compute the flow embedding from the GRU module."
        batch_size, num_frames, channel, w, h = flows.shape
        flow_embeds = self.flow_conv(flows.view(
            batch_size * num_frames, channel, w, h).cuda())
        flow_embeds = flow_embeds.view(batch_size, num_frames, -1)
        _, flow_joint_embeds = self.flow_gru(flow_embeds)
        # Last hidden state remove the output dimension: batchsize * 128
        return flow_joint_embeds[0]

    def compute_visual_embeds(self, crops, full_frames, flows):
        """Compute the all visual embeddings.
        """
        crop_embeds, crop_logits = self._compute_crop_embeds(crops)
        clip_crop_embeds, clip_crop_logits = self._compute_clip_crop_embeds(
            crops)
        clip_frame_embeds, clip_frame_logits = self._compute_clip_frame_embeds(
            full_frames)
        flow_embeds = self._compute_flow_embeds(flows)
        visual_merge_embeds = self.merge_fc(
            torch.cat([crop_embeds,
                       clip_crop_embeds,
                       clip_frame_embeds,
                       flow_embeds], dim=-1))
        merge_logits = self.merge_id_loss(visual_merge_embeds)
        visual_embeds = {"crop_embeds": crop_embeds,
                         "clip_crop_embeds": clip_crop_embeds,
                         "clip_frame_embeds": clip_frame_embeds,
                         "flow_embeds": flow_embeds,
                         "visual_merge_embeds": visual_merge_embeds}
        visual_logits = {"crop_logits": crop_logits,
                         "clip_crop_logits": clip_crop_logits,
                         "clip_frame_logits": clip_frame_logits,
                         "merge_logits": merge_logits}
        return visual_embeds, visual_logits

    def _compute_embeds(self, crops, full_frames, flows, nls):
        """Compute the visual and language embeddings for similary computations

        Args:
            frames (Tensor): num_frames * 3 * W * H
            flows (Tensor): num_frames * 3 * W * H where the last channel is zeros.
            lang_embeds (List[str]): num_nls

        Raises:
            NotImplementedError: [description]

        Returns:
            A dictionary of visual and NL embeddings.
        """
        lang_embeds = self.compute_nl_embeds(nls)
        visual_embeds, visual_logits = self.compute_visual_embeds(
            crops, full_frames, flows)
        embeds = dict()
        if self.training:
            embeds["cls_logits"] = [visual_logits["crop_logits"],
                                    visual_logits["clip_crop_logits"],
                                    visual_logits["clip_frame_logits"],
                                    visual_logits["merge_logits"]]
        embeds.update({"crop_embeds": (visual_embeds["crop_embeds"], lang_embeds["lang_crop_embeds"]),
                       "flow_embeds": (visual_embeds["flow_embeds"], lang_embeds["lang_flow_embeds"]),
                       "clip_frame_embeds": (visual_embeds["clip_frame_embeds"], lang_embeds["lang_clip_frame_embeds"]),
                       "clip_crop_embeds": (visual_embeds["clip_crop_embeds"], lang_embeds["lang_clip_crop_embeds"]),
                       "merge_embeds": (visual_embeds["visual_merge_embeds"], lang_embeds["lang_merge_embeds"])})
        return embeds

    def compute_similarities(self, embeds):
        """Compute the similarities based on the embeds

        Args:
            embeds (Dict): Dictionary of tuples of visual and nl embedding.
            num_nls (int): Number of NL descriptions.
            num_frames (int): Number of frames.
        Returns:
            Dict: dictionary of similarities.
        """
        sims = self._compute_per_module_similarity(embeds)
        return sims["overall_sims"]

    def _compute_per_module_similarity(self, embeds):
        sims = dict()
        for embed_keys in embeds:
            if "embeds" not in embed_keys:
                continue
            visual_embeds, lang_embeds = embeds[embed_keys]
            visual_embeds = torch.nn.functional.normalize(
                visual_embeds, dim=-1)
            lang_embeds = torch.nn.functional.normalize(lang_embeds, dim=-1)
            similarity = torch.matmul(
                self.temperature[embed_keys] * visual_embeds, torch.t(lang_embeds))
            sims[embed_keys + "_sims"] = similarity
        overall_sims = []
        for sim_key in sims:
            overall_sims.append(
                self.sim_weights[sim_key] * sims[sim_key].detach())
        sims["overall_sims"] = torch.mean(
            torch.stack(overall_sims, dim=-1), dim=-1)
        return sims

    def compute_loss(self, batch):
        """Compute ID losses and NCE losses for the input batch.
           For training.
        """
        batch_size = len(batch["random_nl"])
        outputs = self._compute_embeds(
            batch["crop"],
            batch["full_frame"],
            batch["flows"],
            [str(nl) for nl in batch["random_nl"]])
        losses = dict()
        cls_label = batch["target_id"].cuda()
        cls_logit = torch.mean(torch.stack(
            outputs["cls_logits"], dim=-1), dim=-1)
        losses["cls_loss"] = torch.nn.functional.cross_entropy(
            cls_logit, cls_label.cuda())

        sims = self._compute_per_module_similarity(outputs)
        for sim_key in sims:
            sim_i_2_t = sims[sim_key]
            loss_t_2_i = torch.nn.functional.cross_entropy(
                sim_i_2_t, torch.arange(batch_size).cuda())
            sim_t_2_i = sim_i_2_t.t()
            loss_i_2_t = torch.nn.functional.cross_entropy(
                sim_t_2_i, torch.arange(batch_size).cuda())
            losses[sim_key] = (loss_t_2_i+loss_i_2_t)/2
        return losses

    def save_nl_embeds(self, query, query_id, save_path):
        path_to_nl_embeds = os.path.join(
            save_path, "nl_embeds", "%s.pth" % query_id)
        query_embeds = self.compute_nl_embeds(query)
        for key in query_embeds:
            query_embeds[key] = query_embeds[key].cpu()
        torch.save(query_embeds, path_to_nl_embeds)

    def save_visual_embeds(self, track, save_path):
        path_to_visual_embeds = os.path.join(save_path, "visual_embeds", "%s.pth" % str(
            uuid.UUID(bytes=torch.ByteTensor(track["track_id"]).numpy().tobytes())))
        flows = torch.tile(track["flows"][0], [
            len(track["crop"][0]), 1, 1, 1, 1])
        visual_embeds, _ = self.compute_visual_embeds(
            track["crop"][0], track["full_frame"][0], flows)
        for key in visual_embeds:
            visual_embeds[key] = visual_embeds[key].cpu()
        torch.save(visual_embeds, path_to_visual_embeds)

    def inference(self, visual_embeds, query_embeds):
        per_nl_sim = []
        for i in range(len(query_embeds["lang_crop_embeds"])):
            lang_embeds = dict()
            for key in query_embeds:
                lang_embeds[key] = torch.tile(
                    query_embeds[key][i], [len(visual_embeds["crop_embeds"][0]), 1])
            embeddings = {"crop_embeds": (visual_embeds["crop_embeds"], lang_embeds["lang_crop_embeds"]),
                          "flow_embeds": (visual_embeds["flow_embeds"], lang_embeds["lang_flow_embeds"]),
                          "clip_frame_embeds": (visual_embeds["clip_frame_embeds"], lang_embeds["lang_clip_frame_embeds"]),
                          "clip_crop_embeds": (visual_embeds["clip_crop_embeds"], lang_embeds["lang_clip_crop_embeds"]),
                          "merge_embeds": (visual_embeds["visual_merge_embeds"], lang_embeds["lang_merge_embeds"])}
            sims = self._compute_per_module_similarity(embeddings)
            per_nl_sim.append(torch.mean(sims["overall_sims"]))
        return torch.mean(torch.stack(per_nl_sim))

    def forward(self, track, query, query_id, save_path):
        """Forward Pass for inference. See workflow for faster implementation.

        Args:
            track (Dict): A dictionary of tensors.
            query (List[str]): List of NL descriptions in strings.
            query_id ([bytes]): UUID in bytes
            save_path (str): Path to the temporary save directory.

        Returns:
            torch.float32: average overall similarity.
        """
        path_to_visual_embeds = os.path.join(save_path, "visual_embeds", "%s.pth" % str(
            uuid.UUID(bytes=torch.ByteTensor(track["track_id"]).numpy().tobytes())))
        path_to_nl_embeds = os.path.join(
            save_path, "nl_embeds", "%s.pth" % query_id)
        if os.path.isfile(path_to_nl_embeds):
            query_embeds = torch.load(path_to_nl_embeds)
        else:
            query_embeds = self.compute_nl_embeds(query)
            if torch.distributed.get_rank() == 0:
                torch.save(query_embeds, path_to_nl_embeds)
            else:
                time.sleep(5)
        if os.path.isfile(path_to_visual_embeds):
            visual_embeds = torch.load(path_to_visual_embeds)
        else:
            flows = torch.tile(track["flows"][0], [
                               len(track["crop"][0]), 1, 1, 1, 1])
            visual_embeds, _ = self.compute_visual_embeds(
                track["crop"][0], track["full_frame"][0], flows)
            torch.save(visual_embeds, path_to_visual_embeds)
        per_nl_sim = []
        for i in range(len(query)):
            lang_embeds = dict()
            for key in query_embeds:
                lang_embeds[key] = torch.tile(
                    query_embeds[key][i], [len(track["crop"][0]), 1])
            embeddings = {"crop_embeds": (visual_embeds["crop_embeds"], lang_embeds["lang_crop_embeds"]),
                          "flow_embeds": (visual_embeds["flow_embeds"], lang_embeds["lang_flow_embeds"]),
                          "clip_frame_embeds": (visual_embeds["clip_frame_embeds"], lang_embeds["lang_clip_frame_embeds"]),
                          "clip_crop_embeds": (visual_embeds["clip_crop_embeds"], lang_embeds["lang_clip_crop_embeds"]),
                          "merge_embeds": (visual_embeds["visual_merge_embeds"], lang_embeds["lang_merge_embeds"])}
            sims = self._compute_per_module_similarity(embeddings)
            per_nl_sim.append(torch.mean(sims["overall_sims"]))
        return torch.mean(torch.stack(per_nl_sim))
