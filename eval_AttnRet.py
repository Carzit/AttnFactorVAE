import os
import sys
import logging
import argparse
from typing import List, Dict, Tuple, Optional, Literal, Union, Any, Callable

import csv
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler

from dataset import StockDataset, StockSequenceDataset, DataLoader_Preparer
from nets import AttnRet
from loss import MSE_Loss, PearsonCorr, SpearmanCorr
from preparers import Model_AttnRet_Preparer, LoggerPreparer
import utils

class AttnRetEvaluator:
    def __init__(self) -> None:
        
        self.model:AttnRet
        self.pred_eval_func:Union[nn.Module, Callable]
        self.test_loader:DataLoader
        self.subset:str = "test"

        self.model_preparer = Model_AttnRet_Preparer()
        self.dataloader_preparer = DataLoader_Preparer()
        
        self.metric:Literal["MSE", "IC", "RankIC", "ICIR", "RankICIR"]
        self.pred_scores:List[float] = []
        self.y_true_list:List[torch.Tensor] = []
        self.y_pred_list:List[torch.Tensor] = []
        
        self.logger:logging.Logger

        self.device:torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.dtype:torch.dtype = torch.float32

        self.save_folder:str = "."
        self.checkpoints:List[str]
        self.checkpoint_folder:str
        
        self.plotter = utils.Plotter()
        self.plot_index = List[int]

    def set_logger(self, logger:logging.Logger):
        self.logger = logger
        self.model_preparer.set_logger(logger)
        self.dataloader_preparer.set_logger(logger)

    def set_configs(self,
                    device:torch.device,
                    dtype:torch.dtype,
                    subset:str,
                    metric:Literal["MSE", "IC", "RankIC", "ICIR", "RankICIR"],
                    save_folder:str,
                    checkpoints:Optional[List[str]]=None,
                    checkpoint_folder:Optional[str]=None,
                    plot_index:List[int] = [0]):
        if checkpoint_folder:
            checkpoints = [os.path.join(checkpoint_folder, f) for f in os.listdir(checkpoint_folder) if f.endswith((".pt", ".safetensors"))]

        self.device = device
        self.dtype = dtype
        self.subset = subset
        self.metric = metric
        self.save_folder = save_folder
        self.checkpoints = checkpoints
        self.checkpoint_folder = checkpoint_folder
        self.plot_index = plot_index

        os.makedirs(self.save_folder, exist_ok=True)

    def load_configs(self, config_file:str):
        eval_configs = utils.read_configs(config_file=config_file)
        self.model_preparer.set_configs(fundamental_feature_size=eval_configs["Model"]["fundamental_feature_size"],
                                        quantity_price_feature_size=eval_configs["Model"]["quantity_price_feature_size"],
                                        num_gru_layers=eval_configs["Model"]["num_gru_layers"],
                                        gru_hidden_size=eval_configs["Model"]["gru_hidden_size"],
                                        gru_dropout=eval_configs["Model"]["gru_dropout"],
                                        num_fc_layers=eval_configs["Model"]["num_fc_layers"])
        self.dataloader_preparer.set_configs(dataset_path=eval_configs["Dataset"]["dataset_path"],
                                             num_workers=eval_configs["Dataset"]["num_workers"],
                                             shuffle=False,
                                             num_batches_per_epoch=-1)

        self.set_configs(device=utils.str2device(eval_configs["Eval"]["device"]),
                         dtype=utils.str2dtype(eval_configs["Eval"]["dtype"]),
                         metric=eval_configs["Eval"]["metric"],
                         subset=eval_configs["Dataset"]["subset"],
                         checkpoints=eval_configs["Eval"]["checkpoints"],
                         checkpoint_folder=eval_configs["Eval"]["checkpoint_folder"],
                         save_folder=eval_configs["Eval"]["save_folder"],
                         plot_index=eval_configs["Eval"]["plot_index"])
    
    def load_args(self, args:argparse.Namespace | argparse.ArgumentParser):
        if isinstance(args, argparse.ArgumentParser):
            args = args.parse_args()
        self.model_preparer.set_configs(fundamental_feature_size=args.fundamental_feature_size,
                                        quantity_price_feature_size=args.quantity_price_feature_size,
                                        num_gru_layers=args.num_gru_layers,
                                        gru_hidden_size=args.gru_hidden_size,
                                        gru_dropout=args.gru_dropout,
                                        num_fc_layers=args.num_fc_layers)
        self.dataloader_preparer.set_configs(dataset_path=args.dataset_path,
                                             num_workers=args.num_workers,
                                             shuffle=False,
                                             num_batches_per_epoch=-1)

        self.set_configs(device=utils.str2device(args.device),
                         dtype=utils.str2dtype(args.dtype),
                         metric=args.metric,
                         subset=args.subset,
                         checkpoints=args.checkpoints,
                         checkpoint_folder=args.checkpoint_folder,
                         save_folder=args.save_folder,
                         plot_index=args.plot_index)  
        
    def get_configs(self):
        eval_configs = {"device": self.device.type,
                        "dtype": utils.dtype2str(self.dtype),
                        "metric": self.metric,
                        "checkpoints": self.checkpoints,
                        "checkpoint_folder": self.checkpoint_folder,
                        "save_folder": self.save_folder,
                        "plot_index": self.plot_index}
        return eval_configs
    
    def save_configs(self, config_file:Optional[str]=None):
        model_configs = self.model_preparer.get_configs()
        model_configs.pop("checkpoint_path")
        dataset_configs = self.dataloader_preparer.get_configs()
        dataset_configs.pop("shuffle")
        dataset_configs.pop("num_batches_per_epoch")
        configs = {"Model": model_configs,
                   "Dataset": dataset_configs,
                   "Eval": self.get_configs()}
        if not config_file:
            config_file = os.path.join(self.save_folder, "config.json")
        utils.save_configs(config_file=config_file, config_dict=configs)
        
    def prepare(self):
        self.model = self.model_preparer.prepare()
        loaders = self.dataloader_preparer.prepare()
        if self.subset == "train":
            self.test_loader = loaders[0]
        elif self.subset == "eval":
            self.test_loader = loaders[1]
        elif self.subset == "test":
            self.test_loader = loaders[2]

    def eval(self, checkpoint_path):
        utils.load_checkpoint(checkpoint_path=checkpoint_path, model=self.model)
        if self.metric == "MSE":
            self.pred_eval_func = MSE_Loss(scale=1)
        elif self.metric == "IC" or self.metric == "ICIR":
            self.pred_eval_func = PearsonCorr()
        elif self.metric == "RankIC" or self.metric == "RankICIR":
            self.pred_eval_func = SpearmanCorr()
        
        self.pred_scores = []
        self.y_true_list = []
        self.y_pred_list = []
        model = self.model.to(device=self.device, dtype=self.dtype)
        model.eval() # set eval mode to frozen layers like dropout
        with torch.no_grad(): 
            with utils.MeanVarianceAccumulator() as accumulator:
                for batch, (quantity_price_feature, fundamental_feature, label, _) in enumerate(tqdm(self.test_loader, desc=f"{checkpoint_path[checkpoint_path.find('epoch'):checkpoint_path.find('.')]} Eval")):
                    if fundamental_feature.shape[0] <= 2:
                        continue
                    quantity_price_feature = quantity_price_feature.to(device=self.device)
                    fundamental_feature = fundamental_feature.to(device=self.device)
                    label = label.to(device=self.device)
                    y_pred= model(fundamental_feature, quantity_price_feature)
                    score = self.pred_eval_func(y_pred, label)

                    accumulator.accumulate(score.item())
                    self.pred_scores.append(score.item())
                    self.y_true_list.append(label)
                    self.y_pred_list.append(y_pred)
        if self.metric == "MSE" or self.metric == "IC" or self.metric == "RankIC":
            y_pred_score = accumulator.mean()
        elif self.metric == "ICIR" or self.metric == "Rank_ICIR":
            y_pred_score = accumulator.mean() / accumulator.std()
        self.logger.info(f"{checkpoint_path[checkpoint_path.find('epoch'):checkpoint_path.find('.')]} {self.metric} Score: {y_pred_score}")
        with open(os.path.join(self.save_folder, f"{self.metric}_score.csv"), "a") as f:
            writer = csv.writer(f)
            writer.writerow([checkpoint_path, y_pred_score])
    
    def visualize(self, checkpoint_path:str, idx:int=0):
        self.plotter.plot_score(self.pred_scores, metric=self.metric)
        self.plotter.save_fig(os.path.join(self.save_folder, f"{checkpoint_path[checkpoint_path.find('epoch'):checkpoint_path.find('.')]}_{self.metric}_Scores"))
        self.plotter.plot_pred_sample(self.y_true_list, 
                                      self.y_pred_list,
                                      idx=idx)
        self.plotter.save_fig(os.path.join(self.save_folder, f"{checkpoint_path[checkpoint_path.find('epoch'):checkpoint_path.find('.')]}_Trace_{idx}"))

    def evals(self):
        with open(os.path.join(self.save_folder, f"{self.metric}_score.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(['checkpoint', self.metric])
        for checkpoint in self.checkpoints:
            self.eval(checkpoint_path=checkpoint)
            for idx in self.plot_index:
                self.visualize(checkpoint_path=checkpoint, idx=idx)

def parse_args():
    parser = argparse.ArgumentParser(description="AttnRet Evalution")

    parser.add_argument("--log_path", type=str, default="log/eval_AttnRet.log", help="Path of log file. Default `log/eval_AttnRet.log`")

    parser.add_argument("--load_configs", type=str, default=None, help="Path of config file to load. Optional")
    parser.add_argument("--save_configs", type=str, default=None, help="Path of config file to save. Default saved to save_folder as `config.toml`")

    # dataloader config
    parser.add_argument("--dataset_path", type=str, help="Path of dataset .pt file")
    parser.add_argument("--subset", type=str, default="test", help="Subset of dataset, literally `train`, `val` or `test`. Default `test`")
    parser.add_argument("--num_workers", type=int, default=4, help="Num of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. Default 4")
    parser.add_argument("--checkpoints", type=str, nargs="*", default=None, help="Paths of checkpoint")
    parser.add_argument("--checkpoint_folder", type=str, default=None, help="Folder Path of checkpoints")

    # model config
    parser.add_argument("--quantity_price_feature_size", type=int, help="Input size of quantity-price feature")
    parser.add_argument("--fundamental_feature_size", type=int, help="Input size of fundamental feature")
    parser.add_argument("--num_gru_layers", type=int, help="Num of GRU layers in feature extractor.")
    parser.add_argument("--gru_hidden_size", type=int, help="Hidden size of each GRU layer. num_gru_layers * gru_hidden_size i.e. the input size of FactorEncoder and Factor Predictor.")
    parser.add_argument("--num_fc_layers", type=int, default=4, help="Num of full connected layers in MLP")
    
    # eval config
    parser.add_argument("--dtype", type=str, default="FP32", choices=["FP32", "FP64", "FP16", "BF16"], help="Dtype of data and weight tensor. Literally `FP32`, `FP64`, `FP16` or `BF16`. Default `FP32`")
    parser.add_argument("--device", type=str, default="cuda", choices=["auto", "cuda", "cpu"], help="Device to take calculation. Literally `cpu` or `cuda`. Default `cuda`")
    parser.add_argument("--metric", type=str, default="IC", help="Eval metric type, literally `MSE`, `IC`, `Rank_IC`, `ICIR` or `Rank_ICIR`. Default `IC`. ")
    parser.add_argument("--plot_index", type=int, nargs="+", default=[0], help="Stock index to plot Comparison of y_true, y_hat, and y_pred. Default 0")
    parser.add_argument("--save_folder", type=str, default=None, help="Folder to save plot figures")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    logging.getLogger('PIL').setLevel(logging.ERROR)
    logger = LoggerPreparer(name="Eval", 
                            file_level=logging.INFO, 
                            log_file=args.log_path).prepare()
    
    logger.debug(f"Command: {' '.join(sys.argv)}")
    
    evaluator = AttnRetEvaluator()
    evaluator.set_logger(logger=logger)
    if args.load_configs:
        evaluator.load_configs(config_file=args.load_configs)
    else:
        evaluator.load_args(args=args)
    evaluator.prepare()
    evaluator.save_configs()

    logger.info("Eval start...")
    evaluator.evals()
    logger.info("Eval complete.")