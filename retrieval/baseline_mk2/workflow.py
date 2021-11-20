#!/usr/bin/env python
""" Script for training and inference of the V2 model on CityFlow-NL. """

import json
import math
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing
from absl import flags
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from comm.cityflow_nl_dataset_mk2 import (
    CityFlowNLDataset, CityFlowNLInferenceDataset)
from comm.utils import (TqdmToLogger, evaluate_performance, get_logger,
                        prepare_eval_checkpoint_directory,
                        prepare_training_checkpoint_directory)
from retrieval.baseline_mk2.config import get_default_config
from retrieval.baseline_mk2.vrn_mk2 import VehicleRetrievalNetwork
from third_party.lr import TriStageLRScheduler

torch.multiprocessing.set_sharing_strategy('file_system')

flags.DEFINE_integer("num_machines", 1, "Number of machines.")
flags.DEFINE_integer("local_machine", 0,
                     "Master node is 0, worker nodes starts from 1."
                     "Max should be num_machines - 1.")

flags.DEFINE_integer("num_gpus", 2, "Number of GPUs per machines.")
flags.DEFINE_string("config_file", "baseline/default.yaml",
                    "Default Configuration File.")

flags.DEFINE_string("master_ip", "127.0.0.1",
                    "Master node IP for initialization.")
flags.DEFINE_integer("master_port", 12000,
                     "Master node port for initialization.")

flags.DEFINE_integer("num_tracks", -1, "Number of tracks used for training.")

FLAGS = flags.FLAGS


def train_model_on_dataset(rank, train_cfg):
    """For training VRN-MK-II models.

    Args:
        rank (int): GPU rank. Automatically configured by torch.multiprocessing.
        train_cfg (CfgNode): Training configuration CfgNode.
    """
    _logger = get_logger("training")
    torch.cuda.set_device(rank)
    dist_rank = rank + train_cfg.LOCAL_MACHINE * train_cfg.NUM_GPU_PER_MACHINE
    dist.init_process_group(backend="nccl", rank=dist_rank,
                            world_size=train_cfg.WORLD_SIZE,
                            init_method=train_cfg.INIT_METHOD)

    model = VehicleRetrievalNetwork(train_cfg.MODEL).cuda()
    model = DistributedDataParallel(
        model, device_ids=[rank], output_device=rank)
    dataset = CityFlowNLDataset(
        train_cfg.TRAIN_DATA, train_cfg.TRAIN.NUM_FRAMES)
    train_sampler = DistributedSampler(
        dataset, seed=int(datetime.now().timestamp()))
    dataloader = DataLoader(dataset, batch_size=train_cfg.TRAIN.BATCH_SIZE,
                            num_workers=train_cfg.TRAIN.NUM_WORKERS,
                            sampler=train_sampler)
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=train_cfg.TRAIN.LR.INIT_LR)
    if dist_rank == 0:
        prepare_training_checkpoint_directory(train_cfg)
        summary_writer = SummaryWriter(os.path.join(train_cfg.LOG_DIR,
                                                    train_cfg.EXPR_NAME, "summary"))

    global_step = 1
    if train_cfg.TRAIN.RESTORE_FROM != "":
        ckpt = torch.load(train_cfg.TRAIN.RESTORE_FROM,
                          map_location=lambda storage, loc: storage.cpu())
        model.load_state_dict(ckpt["state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step = ckpt["global_step"]
        train_cfg.TRAIN.START_EPOCH = ckpt["epoch"]

    scheduler = TriStageLRScheduler(
        optimizer,
        init_lr=train_cfg.TRAIN.LR.INIT_LR,
        peak_lr=train_cfg.TRAIN.LR.PEAK_LR,
        final_lr=train_cfg.TRAIN.LR.FINAL_LR,
        warmup_steps=train_cfg.TRAIN.LR.WARMUP_STEPS,
        hold_steps=train_cfg.TRAIN.LR.HOLD_STEPS,
        decay_steps=train_cfg.TRAIN.LR.DECAY_STEPS,
        global_step=global_step
    )
    moving_average_loss = []
    for epoch in range(train_cfg.TRAIN.START_EPOCH, train_cfg.TRAIN.EPOCH):
        for data in dataloader:
            optimizer.zero_grad()
            losses = model.module.compute_loss(data)
            loss = torch.sum(torch.stack(list(losses.values())))
            if math.isnan(loss.data.item()):
                _logger.warning("Loss is NAN. Step: %d", global_step)
            elif math.isinf(loss.data.item()):
                _logger.warning("Loss is infinity. Step: %d", global_step)
            elif global_step > 150 and loss.data.item() > train_cfg.TRAIN.LOSS_CLIP_VALUE:
                _logger.warning("Loss %.4f on step %d is greater than predefined threshold %.4f",
                                loss.data.item(), global_step, train_cfg.TRAIN.LOSS_CLIP_VALUE)
            else:
                loss.backward()
                moving_average_loss.append(loss.data.item())
                optimizer.step()
                scheduler.step()

            if global_step % train_cfg.TRAIN.PRINT_FREQ == 0 or global_step < 100:
                if dist_rank == 0:
                    _logger.info("EPOCH\t%d; STEP\t%d; LOSS\t%.4f",
                                 epoch, global_step, loss.data.item())
                    if global_step > 200:
                        summary_writer.add_scalar(
                            "loss/avg_total", np.mean(moving_average_loss), global_step)
                        moving_average_loss = []
                        for key in losses:
                            summary_writer.add_scalar(
                                "loss/%s" % key, losses[key], global_step)
                        summary_writer.add_scalar(
                            "lr", scheduler.get_lr(), global_step)
            if global_step % train_cfg.TRAIN.CHECKPOINT_FREQ == 0:
                if dist_rank == 0:
                    checkpoint_file = os.path.join(train_cfg.LOG_DIR,
                                                   train_cfg.EXPR_NAME, "checkpoints",
                                                   "CKPT-E%d-S%d.pth" % (epoch, global_step))
                    torch.save({"epoch": epoch,
                                "global_step": global_step,
                                "state_dict": model.state_dict(),
                                "optimizer": optimizer.state_dict(), },
                               checkpoint_file)
                    _logger.info("CHECKPOINT SAVED AT: %s",
                                 checkpoint_file)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            global_step += 1
    dist.destroy_process_group()


def _inference_all_queries(rank, eval_cfg, queries):
    """For evaluate VRN-MK-II models on test queries, saving temporary NL embeddings.

    Args:
        rank (int): GPU rank. Automatically configured by torch.multiprocessing.
        eval_cfg (CfgNode): Evaluation configuration CfgNode.
        queries (Dict): Test queries.
    """
    torch.cuda.set_device(rank)
    model = VehicleRetrievalNetwork(eval_cfg.MODEL)
    ckpt = torch.load(eval_cfg.EVAL.RESTORE_FROM,
                      map_location=lambda storage, loc: storage.cpu())
    restore_kv = {key.replace("module.", "", 1): ckpt["state_dict"][key] for key in
                  ckpt["state_dict"].keys()}
    model.load_state_dict(restore_kv, strict=True)
    model = model.cuda()
    model.eval()
    _logger = get_logger("inference_queries")
    temp_path = os.path.join(
        eval_cfg.LOG_DIR, eval_cfg.EXPR_NAME)
    for idx, query_id in enumerate(queries):
        if idx % eval_cfg.WORLD_SIZE != rank:
            continue
        _logger.info("Evaluate query %s on GPU %d", query_id, rank)
        query = queries[query_id]
        with torch.no_grad():
            model.save_nl_embeds(query, query_id, temp_path)


def _inference_all_tracks(rank, eval_cfg):
    """For evaluate VRN-MK-II models on test tracks, saving temporary visual embeddings.

    Args:
        rank (int): GPU rank. Automatically configured by torch.multiprocessing.
        eval_cfg (CfgNode): Evaluation configuration CfgNode.
    """
    torch.cuda.set_device(rank)
    dist_rank = rank + eval_cfg.LOCAL_MACHINE * eval_cfg.NUM_GPU_PER_MACHINE
    dist.init_process_group(backend="nccl", rank=dist_rank,
                            world_size=eval_cfg.WORLD_SIZE,
                            init_method=eval_cfg.INIT_METHOD)
    model = VehicleRetrievalNetwork(eval_cfg.MODEL).cuda()
    model = DistributedDataParallel(
        model, device_ids=[rank], output_device=rank)

    ckpt = torch.load(eval_cfg.EVAL.RESTORE_FROM,
                      map_location=lambda storage, loc: storage.cpu())
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    test_dataset = CityFlowNLInferenceDataset(eval_cfg.TEST_DATA)
    test_sampler = DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=1,
                                 num_workers=eval_cfg.EVAL.NUM_WORKERS,
                                 sampler=test_sampler)
    temp_path = os.path.join(
        eval_cfg.LOG_DIR, eval_cfg.EXPR_NAME)
    pbar = tqdm(total=len(test_dataloader),
                leave=False,
                desc="Eval visual embeds",
                file=TqdmToLogger(),
                mininterval=1,
                maxinterval=100, )
    for data in test_dataloader:
        with torch.no_grad():
            model.module.save_visual_embeds(data, temp_path)
        pbar.update()
    pbar.close()
    dist.destroy_process_group()


def _evaluate_from_files(rank, eval_cfg, queries):
    model = VehicleRetrievalNetwork(eval_cfg.MODEL)
    ckpt = torch.load(eval_cfg.EVAL.RESTORE_FROM,
                      map_location=lambda storage, loc: storage.cpu())
    restore_kv = {key.replace("module.", "", 1): ckpt["state_dict"][key] for key in
                  ckpt["state_dict"].keys()}
    model.load_state_dict(restore_kv, strict=True)
    model.eval()
    _logger = get_logger("inference_sims")
    temp_path = os.path.join(
        eval_cfg.LOG_DIR, eval_cfg.EXPR_NAME)

    with open(eval_cfg.TEST_DATA.JSON_PATH) as track_file:
        tracks = json.load(track_file)
    for idx, query_id in enumerate(queries):
        result = dict()
        if idx % eval_cfg.WORLD_SIZE != rank:
            continue
        path_to_nl_embeds = os.path.join(
            temp_path, "nl_embeds", "%s.pth" % query_id)
        query_embeds = torch.load(path_to_nl_embeds)
        for track_id in tracks:
            _logger.info("Evaluate query %s, track %s, on process %d",
                         query_id, track_id, rank)
            with torch.no_grad():
                path_to_visual_embeds = os.path.join(
                    temp_path, "visual_embeds", "%s.pth" % track_id)
                visual_embeds = torch.load(path_to_visual_embeds)
                sim = model.inference(visual_embeds, query_embeds)
                result[track_id] = sim.data.item()
        with open(os.path.join(temp_path, "logs", "%s.json" % query_id), "w") as f:
            json.dump(result, f)


def _evaluate_from_json_files(eval_cfg):
    _logger = get_logger("evaluation")
    all_query_results = dict()
    with open(eval_cfg.TEST_DATA.QUERIES, "r") as test_q_file:
        test_queries = json.load(test_q_file)
    for query_id in test_queries:
        with open(os.path.join(eval_cfg.LOG_DIR, eval_cfg.EXPR_NAME, "logs", "%s.json" % query_id), "r") as f:
            query_result = json.load(f)
        all_query_results[query_id] = query_result
    with open(eval_cfg.TEST_DATA.GT_PATH, "r") as gt_file:
        test_gt = json.load(gt_file)
    recall_5, recall_10, mrr = evaluate_performance(
        test_gt, all_query_results)
    _logger.info("Recall@5 is %.4f", recall_5)
    _logger.info("Recall@10 is %.4f", recall_10)
    _logger.info("MRR is %.4f", mrr)
    with open(os.path.join(eval_cfg.LOG_DIR, eval_cfg.EXPR_NAME, "test_results.json"), "w") as f:
        json.dump(all_query_results, f)


if __name__ == "__main__":
    FLAGS(sys.argv)
    cfg = get_default_config()
    cfg.merge_from_file(FLAGS.config_file)
    cfg.NUM_GPU_PER_MACHINE = FLAGS.num_gpus
    cfg.NUM_MACHINES = FLAGS.num_machines
    cfg.LOCAL_MACHINE = FLAGS.local_machine
    cfg.WORLD_SIZE = FLAGS.num_machines * FLAGS.num_gpus
    cfg.EXPR_NAME = cfg.EXPR_NAME + "_" + datetime.now().strftime(
        "%m_%d.%H:%M:%S.%f")
    cfg.INIT_METHOD = "tcp://%s:%d" % (FLAGS.master_ip, FLAGS.master_port)
    if cfg.TYPE == "TRAIN":
        cfg.TRAIN.NUM_FRAMES = FLAGS.num_tracks
        torch.multiprocessing.spawn(train_model_on_dataset, args=(cfg,),
                                    nprocs=cfg.NUM_GPU_PER_MACHINE, join=True)
    elif cfg.TYPE == "EVAL":
        with open(cfg.TEST_DATA.QUERIES, "r") as f:
            test_queries = json.load(f)
        if os.path.isdir(cfg.EVAL.CONTINUE):
            files = os.listdir(os.path.join(cfg.EVAL.CONTINUE, "logs"))
            for q in files:
                del test_queries[q.split(".")[0]]
            cfg.EXPR_NAME = cfg.EVAL.CONTINUE.split("/")[-1]
        else:
            prepare_eval_checkpoint_directory(cfg)
        torch.multiprocessing.spawn(_inference_all_queries, args=(cfg, test_queries, ),
                                    nprocs=cfg.NUM_GPU_PER_MACHINE, join=True)
        torch.multiprocessing.spawn(_inference_all_tracks, args=(cfg, ),
                                    nprocs=cfg.NUM_GPU_PER_MACHINE, join=True)
        # Run 20 processes.
        torch.multiprocessing.spawn(_evaluate_from_files, args=(cfg, test_queries, ),
                                    nprocs=20, join=True)
        _evaluate_from_json_files(cfg)
