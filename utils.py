import os
import ast
import argparse
from typing import Tuple, Literal, Union, Optional, List, Dict, Any

import json
import toml
import yaml

import pandas as pd
import numpy as np
import torch
from safetensors.torch import load_file, save_file

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
        raise argparse.ArgumentTypeError(f"Unexpected dtype `{dtype}`. dtype must be `FP32`, `FP64`, `FP16` or `BF16`.")
    
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
        raise argparse.ArgumentTypeError(f"Unexpected dtype `{dtype}`. dtype must be `FP32`, `FP64`, `FP16` or `BF16`.")
    
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
    
@torch.no_grad()
def drop_nan_inf3(a:torch.Tensor, b:torch.Tensor, c:torch.Tensor) -> Tuple[torch.Tensor]:
    a_valid_mask = torch.all(~torch.isnan(a) & ~torch.isinf(a), dim=(0, 2))
    b_valid_mask = torch.all(~torch.isnan(b) & ~torch.isinf(b), dim=1)
    c_valid_mask = ~torch.isnan(c) & ~torch.isinf(c)
    
    valid_indices = a_valid_mask & b_valid_mask & c_valid_mask
    
    a_filtered = a[:, valid_indices, :]
    b_filtered = b[valid_indices, :]
    c_filtered = c[valid_indices]
    
    return a_filtered, b_filtered, c_filtered, valid_indices

@torch.no_grad()
def loose_drop_nan_inf3(a:torch.Tensor, b:torch.Tensor, c:torch.Tensor) -> Tuple[torch.Tensor]:
    a_valid_mask = torch.any(~torch.isnan(a) & ~torch.isinf(a), dim=(0, 2))
    b_valid_mask = torch.any(~torch.isnan(b) & ~torch.isinf(b), dim=1)
    c_valid_mask = ~torch.isnan(c) & ~torch.isinf(c)
    
    valid_indices = a_valid_mask & b_valid_mask & c_valid_mask
    
    a_filtered = a[:, valid_indices, :]
    b_filtered = b[valid_indices, :]
    c_filtered = c[valid_indices]
    
    return a_filtered, b_filtered, c_filtered, valid_indices

@torch.no_grad()
def drop_nan_inf2(a:torch.Tensor, c:torch.Tensor) -> Tuple[torch.Tensor]:
    a_valid_mask = torch.all(~torch.isnan(a) & ~torch.isinf(a), dim=(0, 2))
    c_valid_mask = ~torch.isnan(c) & ~torch.isinf(c)
    
    valid_indices = a_valid_mask & b_valid_mask & c_valid_mask
    
    a_filtered = a[:, valid_indices, :]
    b_filtered = b[valid_indices, :]
    c_filtered = c[valid_indices]
    
    return a_filtered, b_filtered, c_filtered, valid_indices

@torch.no_grad()
def loose_drop_nan_inf3(a:torch.Tensor, c:torch.Tensor) -> Tuple[torch.Tensor]:
    a_valid_mask = torch.any(~torch.isnan(a) & ~torch.isinf(a), dim=(0, 2))
    b_valid_mask = torch.any(~torch.isnan(b) & ~torch.isinf(b), dim=1)
    c_valid_mask = ~torch.isnan(c) & ~torch.isinf(c)
    
    valid_indices = a_valid_mask & b_valid_mask & c_valid_mask
    
    a_filtered = a[:, valid_indices, :]
    b_filtered = b[valid_indices, :]
    c_filtered = c[valid_indices]
    
    return a_filtered, b_filtered, c_filtered, valid_indices
    

