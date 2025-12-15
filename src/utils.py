import random
import os
import numpy as np
import torch
import lgging
import yaml
import json
from types import SimpleNamespace

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

def init_logger(log_file=f'train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.propagate = False
    return logger

IS_KAGGLE_ENV = sum(['KAGGLE' in k for k in os.environ]) > 0
IS_KAGGLE_SUBMISSION = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
suffix = datetime.now().strftime("%Y%m%d%H%M%S")[-1*10:]

def load_config(yaml_path):
    def _recursive_namespace(d):
        if isinstance(d, dict):
            for k, v in d.items():
                d[k] = _recursive_namespace(v)
            return SimpleNamespace(**d)
        elif isinstance(d, list):
            return [_recursive_namespace(item) for item in d]
        else:
            return d
    with open(yaml_path, 'r', encoding = 'utf-8') as f:
        cfg_dict = yaml.safe_load(f)

    cfg = _recursive_namespace(cfg_dict)
