"""YACS based configs for VRN MK2."""
# Configs for train/evaluate the baseline retrieval model.

from fvcore.common.config import CfgNode as CN

from comm.cityflow_nl_dataset_mk2 import get_default_data_config

_C = CN()

_C.VERSION = 2

_C.EXPR_NAME = "EXPR"
_C.LOG_DIR = "experiments/"
# Workflow type, can be either TRAIN or EVAL.
_C.TYPE = "TRAIN"

# CityFlow-NL related configurations.
_C.TRAIN_DATA = get_default_data_config()

_C.TEST_DATA = get_default_data_config()
_C.TEST_DATA.SPLIT = "test"
_C.TEST_DATA.JSON_PATH = "data/test-tracks.json"
_C.TEST_DATA.QUERIES = "data/test-queries_nlpaug.json"
_C.TEST_DATA.GT_PATH = "data/test-gt.json"
_C.TEST_DATA.POSITIVE_THRESHOLD = 0.
_C.TEST_DATA.FRAME_PER_TRACK=16

_C.VAL_DATA = get_default_data_config()
_C.VAL_DATA.SPLIT = "test"
_C.VAL_DATA.JSON_PATH = "data/val-tracks.json"
_C.VAL_DATA.QUERIES = "data/val-queries.json"
_C.VAL_DATA.GT_PATH = "data/val-gt.json"
_C.VAL_DATA.POSITIVE_THRESHOLD = 0.
_C.VAL_DATA.FRAME_PER_TRACK=16

# Model specific configurations.
_C.MODEL = CN()
# Checkpoints for BERT and RESNET
_C.MODEL.BERT_CHECKPOINT = "checkpoints/bert-base-uncased/"
_C.MODEL.CLIP_CHECKPOINT = "checkpoints/clip-vit-base-patch32/"
_C.MODEL.OUTPUT_SIZE = 256
_C.MODEL.NUM_CLASS = 667  # 666 unique targets + 0 for negative class.
_C.MODEL.USE_CAR_ID_LOSS = True
_C.MODEL.USE_MOTION_ID_LOSS = True
_C.MODEL.USE_SHARE_ID_LOSS = True
_C.MODEL.USE_COSINE_SIMILARITY = True
_C.MODEL.USE_FULL_FRAME_ID_LOSS = True
_C.MODEL.USE_CLIP_ID_LOSS = True

# Training configurations
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 1
_C.TRAIN.EPOCH = 2000
_C.TRAIN.NUM_WORKERS = 8
_C.TRAIN.PRINT_FREQ = 50
_C.TRAIN.CHECKPOINT_FREQ = 1000
_C.TRAIN.LOSS_CLIP_VALUE = 200.
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.RESTORE_FROM = ""

_C.TRAIN.LR = CN()
_C.TRAIN.LR.INIT_LR = 1E-5
_C.TRAIN.LR.PEAK_LR = 1E-3
_C.TRAIN.LR.FINAL_LR = 1E-5
_C.TRAIN.LR.WARMUP_STEPS = 2000
_C.TRAIN.LR.HOLD_STEPS = 10000
_C.TRAIN.LR.DECAY_STEPS = 50000

_C.EVAL = CN()
_C.EVAL.RESTORE_FROM = ""
_C.EVAL.NUM_WORKERS = 8
_C.EVAL.CONTINUE = ""

def get_default_config():
    """[summary]

    Returns:
        yacs.CfgNode: A default VRN MK2 config.
    """
    return _C.clone()
