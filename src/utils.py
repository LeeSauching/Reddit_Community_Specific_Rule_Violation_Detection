import random
import os
import numpy as np
import torch
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

def init_logger(log_file):
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

def is_kaggle_env():
    return 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

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

    on_kaggle = is_kaggle_env()

    if on_kaggle:
        base_input = "/kaggle/input"
        base_output = "kaggle/working"
        
        if 'input_dir' in cfg_dict:
            cfg_dict['input_dir'] = os.path.join(base_input, cfg_dict.get('comp_name', ''))

        if 'base_model_path' in cfg_dict:
            cfg_dict['base_model_path'] = os.path.join(base_input, fg_dict['base_model_path'])

        if 'output_dir' in cfg_dict:
            cfg_dict['output_dir'] = base_output
    else:
        if 'base_model_path' in cfg_dict:
            cfg_dict['base_model_path'] = os.path.join("./input", cfg_dict['base_model_path'])
    
    cfg = _recursive_namespace(cfg_dict)
    print(f"Loaded config from {yaml_path}")
    print(json.dumps(cfg_dict, indent=2, ensure_ascii=False))
    
    return cfg
