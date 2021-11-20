# Configs for train/evaluate the baseline retrieval model.

from yacs.config import CfgNode as CN

from comm.cityflow_nl_dataset import get_default_data_config

_C = CN()

_C.EXPR_NAME = "EXPR"
_C.LOG_DIR = "experiments/"
# Workflow type, can be either TRAIN or EVAL.
_C.TYPE = "TRAIN"

# CityFlow-NL related configurations.
_C.DATA = get_default_data_config()

# Model specific configurations.
_C.MODEL = CN()
# Checkpoints for BERT and RESNET
_C.MODEL.BERT_CHECKPOINT = "checkpoints/bert-base-uncased/"
_C.MODEL.RESNET_CHECKPOINT = "checkpoints/resnet50-19c8e357.pth"
_C.MODEL.OUTPUT_SIZE = 256

# Training configurations
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 1
_C.TRAIN.EPOCH = 20
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.NUM_WORKERS = 2
_C.TRAIN.PRINT_FREQ = 250
_C.TRAIN.LOSS_CLIP_VALUE = 10.
_C.TRAIN.LR = CN()
_C.TRAIN.LR.MOMENTUM = 0.9
_C.TRAIN.LR.WEIGHT_DECAY = 0.5
_C.TRAIN.LR.BASE_LR = 0.01
_C.TRAIN.LR.STEP_SIZE = 2

# Evaluation configurations
_C.EVAL = CN()
_C.EVAL.RESTORE_FROM = "experiments/checkpoints/CKPT-E9-S28485.pth"
_C.EVAL.QUERY_JSON_PATH = "data/test-queries.json"
_C.EVAL.BATCH_SIZE = 1
_C.EVAL.NUM_WORKERS = 4
_C.EVAL.CONTINUE = ""
_C.EVAL.MTSC_TRACKER = ""


def get_default_config():
    return _C.clone()
