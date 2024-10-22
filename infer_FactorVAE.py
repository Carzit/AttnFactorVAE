import os
import sys
import logging
import argparse
from numbers import Number
from typing import List, Dict, Tuple, Optional, Literal, Union, Any, Callable

from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler

from dataset import StockDataset, StockSequenceCatDataset, DataLoader_Preparer
from nets import FactorVAE
from preparers import Model_FactorVAE_Preparer, LoggerPreparer
import utils


class FactorVAEInfer:
    def __init__(self) -> None:
        
        self.model:FactorVAE
        self.test_loader:DataLoader
        self.subset:str = "test"

        self.model_preparer = Model_FactorVAE_Preparer()
        self.dataloader_preparer = DataLoader_Preparer()
        
        self.dates:List[str]
        self.stock_codes:List[str]
        self.seq_len:int
        
        self.logger:logging.Logger

        self.device:torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.dtype:torch.dtype = torch.float32

        self.save_folder:str = "."
        self.save_format:Literal["csv", "pkl", "parquet", "feather"] = "parquet"

    def set_logger(self, logger:logging.Logger):
        self.logger = logger
        self.model_preparer.set_logger(logger)
        self.dataloader_preparer.set_logger(logger)

    def set_configs(self,
                    device:torch.device,
                    dtype:torch.dtype,
                    subset:str,
                    save_format:Literal["csv", "pkl", "parquet", "feather"],
                    save_folder:str):

        self.device = device
        self.dtype = dtype
        self.subset = subset
        self.save_format = save_format
        self.save_folder = save_folder

        os.makedirs(self.save_folder, exist_ok=True)

    def load_configs(self, config_file:str):
        infer_configs = utils.read_configs(config_file=config_file)
        self.model_preparer.load_configs(config_file=config_file)
        self.dataloader_preparer.set_configs(dataset_path=infer_configs["Dataset"]["dataset_path"],
                                             num_workers=infer_configs["Dataset"]["num_workers"],
                                             shuffle=False,
                                             num_batches_per_epoch=-1)
        self.set_configs(device=utils.str2device(infer_configs["Infer"]["device"]),
                         dtype=utils.str2dtype(infer_configs["Infer"]["dtype"]),
                         subset=infer_configs["Dataset"]["subset"],
                         save_format=infer_configs["Infer"]["save_format"],
                         save_folder=infer_configs["Infer"]["save_folder"])
    
    def load_args(self, args:argparse.Namespace | argparse.ArgumentParser):
        if isinstance(args, argparse.ArgumentParser):
            args = args.parse_args()
        self.model_preparer.load_args(args=args)
        self.dataloader_preparer.set_configs(dataset_path=args.dataset_path,
                                             num_workers=args.num_workers,
                                             shuffle=False,
                                             num_batches_per_epoch=-1)

        self.set_configs(device=utils.str2device(args.device),
                         dtype=utils.str2dtype(args.dtype),
                         subset=args.subset,
                         save_format=args.save_format,
                         save_folder=args.save_folder)  
        
    def get_configs(self):
        eval_configs = {"device": self.device.type,
                        "dtype": utils.dtype2str(self.dtype),
                        "save_format": self.save_format,
                        "save_folder": self.save_folder}
        return eval_configs
    
    def save_configs(self, config_file:Optional[str]=None):
        model_configs = self.model_preparer.get_configs()
        dataset_configs = self.dataloader_preparer.get_configs()
        dataset_configs["subset"] = self.subset
        dataset_configs.pop("shuffle")
        dataset_configs.pop("num_batches_per_epoch")
        configs = {"Model": model_configs,
                   "Dataset": dataset_configs,
                   "Infer": self.get_configs()}
        if not config_file:
            config_file = os.path.join(self.save_folder, "config.json")
        utils.save_configs(config_file=config_file, config_dict=configs)
        
    def prepare(self):
        self.model = self.model_preparer.prepare()
        loaders = self.dataloader_preparer.prepare()
        if self.subset == "train":
            self.test_loader = loaders[0]
        elif self.subset == "val":
            self.test_loader = loaders[1]
        elif self.subset == "test":
            self.test_loader = loaders[2]
        self.stock_codes = self.test_loader.dataset.stock_codes
        self.dates = self.test_loader.dataset.dates[self.test_loader.dataset.seq_len-1:]
    
    def infer(self):
        model = self.model.to(device=self.device, dtype=self.dtype)
        model.eval() # set eval mode to frozen layers like dropout
        with torch.no_grad(): 
            for batch, (feature, label, valid_indices) in enumerate(tqdm(self.test_loader, desc="Infer")):
                if feature.shape[0] <= 2:
                    continue
                feature = feature.to(device=self.device, dtype=self.dtype)
                label = label.to(device=self.device, dtype=self.dtype)
                valid_indices = valid_indices.to(device=self.device)
                y_pred, *_ = model.predict(feature)

                date = self.dates[batch]
                df = self.sparse_fill(y_pred, valid_indices)
                utils.save_dataframe(df=df, path=os.path.join(self.save_folder, date), format=self.save_format) 

        results_df = self.aggregate()
        utils.save_dataframe(df=results_df, path=os.path.join(self.save_folder, "results"), format=self.save_format)
    
    def sparse_fill(self, predictions:torch.Tensor, valid_indices:torch.Tensor) -> None:
        num_stocks = len(self.stock_codes)
        full_predictions = torch.full((num_stocks,), fill_value=np.nan, device=self.device)
        full_predictions[valid_indices] = predictions
        df = pd.DataFrame({
            'stock_code': self.stock_codes, 
            'prediction': full_predictions.cpu().numpy()
        })
        return df
        
    def aggregate(self) -> pd.DataFrame:
        all_predictions = []
        for date in tqdm(self.dates, desc="Merge results"):
            file_path = os.path.join(self.save_folder, f"{date}.{self.save_format}")
            df = utils.load_dataframe(path=file_path, format=self.save_format)
            df = df.set_index('stock_code')     
            all_predictions.append(df['prediction'].rename(date))
        result_df = pd.concat(all_predictions, axis=1).T
        return result_df


def parse_args():
    parser = argparse.ArgumentParser(description="FactorVAE Inference")

    parser.add_argument("--log_path", type=str, default="log/infer_FactorVAE.log", help="Path of log file. Default `log/infer_FactorVAE.log`")

    parser.add_argument("--load_configs", type=str, default=None, help="Path of config file to load. Optional")
    parser.add_argument("--save_configs", type=str, default=None, help="Path of config file to save. Default saved to save_folder as `config.json`")

    # dataloader config
    parser.add_argument("--dataset_path", type=str, help="Path of dataset .pt file")
    parser.add_argument("--subset", type=str, default="test", help="Subset of dataset, literally `train`, `val` or `test`. Default `test`")
    parser.add_argument("--num_workers", type=int, default=4, help="Num of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. Default 4")
    
    # model config
    parser.add_argument("--feature_size", type=int, help="Input size of features")
    parser.add_argument("--num_gru_layers", type=int, help="Num of GRU layers in feature extractor.")
    parser.add_argument("--gru_hidden_size", type=int, help="Hidden size of each GRU layer. num_gru_layers * gru_hidden_size i.e. the input size of FactorEncoder and Factor Predictor.")
    parser.add_argument("--hidden_size", type=int, help="Hidden size of FactorVAE(Encoder, Pedictor and Decoder), i.e. num of portfolios.")
    parser.add_argument("--latent_size", type=int, help="Latent size of FactorVAE(Encoder, Pedictor and Decoder), i.e. num of factors.")
    parser.add_argument("--std_activation", type=str, default="exp", help="Activation function for standard deviation calculation, literally `exp` or `softplus`. Default `exp`")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Folder Path of checkpoint")
    
    # infer config
    parser.add_argument("--dtype", type=str, default="FP32", choices=["FP32", "FP64", "FP16", "BF16"], help="Dtype of data and weight tensor. Literally `FP32`, `FP64`, `FP16` or `BF16`. Default `FP32`")
    parser.add_argument("--device", type=str, default="cuda", choices=["auto", "cuda", "cpu"], help="Device to take calculation. Literally `cpu` or `cuda`. Default `cuda`")
    parser.add_argument("--save_folder", type=str, default=None, help="Folder to save plot figures")
    parser.add_argument("--save_format", type=str, default="pkl", help="File format to save, literally `csv`, `pkl`, `parquet` or `feather`. Default `pkl`")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    logger = LoggerPreparer(name="Infer", 
                            file_level=logging.INFO, 
                            log_file=args.log_path).prepare()
    
    logger.debug(f"Command: {' '.join(sys.argv)}")
    
    infer = FactorVAEInfer()
    infer.set_logger(logger=logger)
    if args.load_configs:
        infer.load_configs(config_file=args.load_configs)
    else:
        infer.load_args(args=args)
    infer.prepare()
    infer.save_configs()

    logger.info("Infer start...")
    infer.infer()
    logger.info("Infer complete.")
        




                    
            
                

    

