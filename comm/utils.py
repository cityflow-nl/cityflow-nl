# Logging utilities.

import io
import logging
import os

import colorlog
import torch
import torch.distributed as dist


class TqdmToLogger(io.StringIO):
    """Progress logger based on tqdm
    """
    logger = None
    level = None
    buf = ""

    def __init__(self):
        super(TqdmToLogger, self).__init__()
        self.logger = get_logger("tqdm")

    def write(self, buf):
        self.buf = buf.strip("\r\n\t ")

    def flush(self):
        self.logger.info(self.buf)


def get_logger(logger_name="default", debug=False, save_to_dir=None):
    if debug:
        log_format = (
            "%(asctime)s - "
            "%(levelname)s : "
            "%(name)s - "
            "%(pathname)s[%(lineno)d]:"
            "%(funcName)s - "
            "%(message)s"
        )
    else:
        log_format = (
            "%(asctime)s - "
            "%(levelname)s : "
            "%(name)s - "
            "%(message)s"
        )
    bold_seq = "\033[1m"
    colorlog_format = f"{bold_seq} %(log_color)s {log_format}"
    colorlog.basicConfig(format=colorlog_format, datefmt="%y-%m-%d %H:%M:%S")
    logger = logging.getLogger(logger_name)

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    try:
        rank = dist.get_rank()
    except RuntimeError:
        rank = 0
    if rank != 0:
        logger.setLevel(logging.WARNING)

    if save_to_dir is not None:
        f_h = logging.FileHandler(os.path.join(
            save_to_dir, "log", "debug.log"))
        f_h.setLevel(logging.DEBUG)
        formatter = logging.Formatter(log_format)
        f_h.setFormatter(formatter)
        logger.addHandler(f_h)

        f_h = logging.FileHandler(
            os.path.join(save_to_dir, "log", "warning.log"))
        f_h.setLevel(logging.WARNING)
        formatter = logging.Formatter(log_format)
        f_h.setFormatter(formatter)
        logger.addHandler(f_h)

        f_h = logging.FileHandler(os.path.join(
            save_to_dir, "log", "error.log"))
        f_h.setLevel(logging.ERROR)
        formatter = logging.Formatter(log_format)
        f_h.setFormatter(formatter)
        logger.addHandler(f_h)

    return logger


def prepare_training_checkpoint_directory(train_cfg):
    prepare_eval_checkpoint_directory(train_cfg)
    os.makedirs(os.path.join(train_cfg.LOG_DIR,
                             train_cfg.EXPR_NAME, "summary"), exist_ok=True)
    os.makedirs(os.path.join(train_cfg.LOG_DIR,
                             train_cfg.EXPR_NAME, "checkpoints"), exist_ok=True)


def prepare_eval_checkpoint_directory(eval_cfg):
    if not os.path.exists(os.path.join(eval_cfg.LOG_DIR, eval_cfg.EXPR_NAME)):
        os.makedirs(os.path.join(eval_cfg.LOG_DIR,
                                 eval_cfg.EXPR_NAME))
    with open(os.path.join(eval_cfg.LOG_DIR,
                           eval_cfg.EXPR_NAME,
                           "config.yaml"), "w") as f:
        f.write(eval_cfg.dump())
    os.makedirs(os.path.join(eval_cfg.LOG_DIR,
                             eval_cfg.EXPR_NAME, "visual_embeds"), exist_ok=True)
    os.makedirs(os.path.join(eval_cfg.LOG_DIR,
                             eval_cfg.EXPR_NAME, "nl_embeds"), exist_ok=True)
    os.makedirs(os.path.join(eval_cfg.LOG_DIR,
                             eval_cfg.EXPR_NAME, "logs"), exist_ok=True)    


def gather_all(tensors):
    """
    All gathers the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all gather across all processes in
        all machines.
    """

    gather_list = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [
            torch.zeros_like(tensor) for _ in range(world_size)
        ]
        dist.all_gather(tensor_placeholder, tensor, async_op=False)
        gather_list.append(tensor_placeholder)
    return gather_list


def evaluate_performance(ground_truth, similarity_dict):
    recall_5 = 0
    recall_10 = 0
    mrr = 0
    for query in ground_truth:
        try:
            query_result = similarity_dict[query]
            result = sorted(query_result, key=query_result.get, reverse=True)
            target = ground_truth[query]
            rank = result.index(target)
        except Exception:
            rank = len(ground_truth)
        if rank < 10:
            recall_10 += 1
        if rank < 5:
            recall_5 += 1
        mrr += 1.0 / (rank + 1)
    recall_5 /= len(ground_truth)
    recall_10 /= len(ground_truth)
    mrr /= len(ground_truth)
    return recall_5, recall_10, mrr
