import os
import ast
import logging
import argparse
from typing import Tuple, Literal, Union, Optional, List, Dict, Any
import math
import json
import toml
import yaml

import pandas as pd
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from matplotlib import pyplot as plt

def str2bool(value:Union[bool, str]):
    if isinstance(value, bool):
        return value
    elif isinstance(value, str):
        if value.lower() in {'false', '0', 'no', 'n', 'f'}:
            return False
        elif value.lower() in {'true', '1', 'yes', 'y', 't'}:
            return True
    else:
        raise argparse.ArgumentTypeError(f'Boolean value or bool like string expected. Get unexpected value {value}, whose type is {type(value)}')

def str2dict(args_list):
    result_dict = {}
    if args_list is not None and len(args_list) > 0:
        for arg in args_list:
            key, value = arg.split("=", 1)  # 使用 1 限制分割次数，避免错误处理包含 '=' 的值
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError): # 如果 literal_eval 失败，就把 value 当作字符串处理
                pass
            result_dict[key] = value
    return result_dict

def str2dtype(dtype:Literal["FP32", "FP64", "FP16", "BF16"]) -> torch.dtype:
    if dtype == "FP32":
        return torch.float32
    elif dtype == "FP64":
        return torch.float64
    elif dtype == "FP16":
        return torch.float16
    elif dtype == "BF16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unexpected dtype `{dtype}`. dtype must be `FP32`, `FP64`, `FP16` or `BF16`.")
    
def dtype2str(dtype:torch.dtype) -> str:
    if dtype == torch.float32:
        return "FP32"
    elif dtype == torch.float64:
        return "FP64"
    elif dtype == torch.float16:
        return "FP16"
    elif dtype == torch.bfloat16:
        return "BF16"
    else:
        raise ValueError(f"Unexpected dtype `{dtype}`. dtype must be `torch.float32`, `torch.float64`, `torch.float16` or `torch.bfloat16`.")
    
def str2dtype_np(dtype:Literal["FP32", "FP64", "FP16", "BF16"]) -> torch.dtype:
    if dtype == "FP32":
        return np.float32
    elif dtype == "FP64":
        return np.float64
    elif dtype == "FP16":
        return np.float16
    else:
        raise ValueError(f"Unexpected dtype `{dtype}`. dtype must be `FP32`, `FP64`, `FP16` or `BF16`.")
    
def dtype2str_np(dtype:torch.dtype) -> str:
    if dtype == np.float32:
        return "FP32"
    elif dtype == np.float64:
        return "FP64"
    elif dtype == np.float16:
        return "FP16"
    else:
        raise ValueError(f"Unexpected dtype `{dtype}`. dtype must be `numpy.float32`, `numpy.float64` or `numpy.float16`.")
    
def str2device(device:Literal["auto", "cpu", "cuda"]) -> torch.device:
    if device == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    elif device.lower() == "cpu":
        return torch.device("cpu")
    elif device.lower() == "cuda":
        return torch.device("cuda")
    else:
        raise argparse.ArgumentTypeError(f"Unexpected device `{device}`. dtype must be `cuda`, `cpu` or `auto`.")

def save_dataframe(df:pd.DataFrame, path:str, format:Literal["csv", "pkl", "parquet", "feather"]="pkl") -> None:
    if not path.endswith(format):
        path += f".{format}"
    if format == "csv":
        df.to_csv(path)
    elif format ==  "pkl":
        df.to_pickle(path)
    elif format == "parquet":
        df.to_parquet(path)
    elif format == "feather":
        df.to_feather(path)
    else:
        raise NotImplementedError()

def load_dataframe(path:str, format:Literal["csv", "pkl", "parquet", "feather"]="pkl") -> pd.DataFrame:
    if format == "csv":
        df = pd.read_csv(path, index_col=0)
    elif format ==  "pkl":
        df = pd.read_pickle(path)
    elif format == "parquet":
        df = pd.read_parquet(path)
    elif format == "feather":
        df = pd.read_feather(path)
    else:
        raise NotImplementedError()
    return df

def save_checkpoint(model:torch.nn.Module, save_folder:str, save_name:str, save_format:Literal[".pt",".safetensors"]=".pt"):
    save_path = os.path.join(save_folder, save_name+save_format)
    if save_format == ".pt":
        torch.save(model.state_dict(), save_path)
    elif save_format == ".safetensors":
        save_file(model.state_dict(), save_path)
    else:
        raise ValueError(f"Unrecognized file format`{save_format}`")

def load_checkpoint(model:torch.nn.Module, checkpoint_path:str):
    if checkpoint_path.endswith(".pt"):
        model.load_state_dict(torch.load(checkpoint_path, weights_only=False))
    elif checkpoint_path.endswith(".safetensors"):
        model.load_state_dict(load_file(checkpoint_path))
    else:
        raise ValueError(f"Unrecognized model weights file `{checkpoint_path}`")

def check(tensor:torch.Tensor):
    return torch.any(torch.isnan(tensor) | torch.isinf(tensor))

def check_attr_dict_match(obj, dic:Dict, names:List[str]):
    for name in names:
        assert hasattr(obj, name) and name in dic
        assert getattr(obj, name) == dic[name]

def read_configs(config_file:str) -> Dict:
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file `{config_file}` not found.")
    file_ext = os.path.splitext(config_file)[1].lower()

    with open(config_file, 'r') as f:
        if file_ext == '.json':
            config_dict = json.load(f)
        elif file_ext == '.toml':
            config_dict = toml.load(f)
        elif file_ext in ['.yaml', '.yml']:
            config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {file_ext}")
    return config_dict

def save_configs(config_file:str, config_dict:Dict):
    file_ext = os.path.splitext(config_file)[1].lower()
    with open(config_file, 'w') as f:
        if file_ext == '.json':
            json.dump(config_dict, f, indent=4)
        elif file_ext == '.toml':
            toml.dump(config_dict, f)
        elif file_ext in ['.yaml', '.yml']:
            yaml.safe_dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config file format: {file_ext}")

def find_common_root(dirs:str):
    if not dirs:
        raise ValueError("The list of directories is empty.")
    try:
        common_root = os.path.commonpath(dirs)
        if not any([os.path.commonprefix([dir_, common_root]) == common_root for dir_ in dirs]):
            raise RuntimeError("No common root directory found.")
        return common_root
    except ValueError:
        raise RuntimeError("No common root directory found.")

class MeanVarianceAccumulator:
    def __init__(self):
        self._n = 0        # 计数器
        self._mean = 0.0   # 均值
        self._m2 = 0.0     # 用于计算方差的中间量

    def accumulate(self, x):
        # 更新计数
        self._n += 1

        # 计算新的均值
        delta = x - self._mean
        self._mean += delta / self._n

        # 更新M2，用于方差计算
        delta2 = x - self._mean
        self._m2 += delta * delta2

    def var(self, ddof:int=0):
        if self._n < 2:
            return float('nan')  # 当样本数小于2时，方差无定义
        return self._m2 / (self._n - ddof)  # 使用无偏估计
    
    def std(self, ddof:int=0):
        return math.sqrt(self.var(ddof=ddof))

    def mean(self):
        return self._mean
    
    def __enter__(self):
        """进入上下文时，初始化/重置计算"""
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """离开上下文时存储最终的均值和方差"""
        pass

class Plotter:
    def __init__(self) -> None:
        self.logger:logging.Logger = logging

    def set_logger(self, logger:logging.Logger):
        if not logger:
            logger = logging
        self.logger = logger
    
    def plot_score(self, pred_scores:List[float], metric:str):
        plt.figure(figsize=(10, 6))
        plt.plot(pred_scores, marker='', color="b")
        plt.title(f'{metric} Scores')
        plt.xlabel('Date')
        plt.ylabel('Score')
    
    def plot_pred_sample(self, y_true_list:List[float], y_pred_list:List[float], y_hat_list:Optional[List[float]]=None, idx=0):
        y_true_list = [y_true[idx].item() for y_true in y_true_list]
        y_pred_list = [y_pred[idx].item() for y_pred in y_pred_list]

        plt.figure(figsize=(10, 6))
        plt.plot(y_true_list, label='y true', marker='', color="g")
        plt.plot(y_pred_list, label='y pred', marker='', color="r")

        if y_hat_list:
            y_hat_list = [y_hat[idx].item() for y_hat in y_hat_list]
            plt.plot(y_hat_list, label='y hat', marker='', color="b")

        plt.legend()
        plt.title('Comparison of y_true and y_pred')
        plt.xlabel('Date')
        plt.ylabel('Value')

    def save_fig(self, filename:str):
        if not filename.endswith(".png"):
            filename = filename + ".png"
        plt.savefig(filename)
        plt.close()
    
