import os
import sys
import datetime
import logging
import argparse
from numbers import Number
from typing import List, Dict, Tuple, Optional, Literal, Union, Any, Callable

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.tensorboard.writer import SummaryWriter


from dataset import DataLoader_Preparer, StockSequenceDataset, StockDataset
from nets import AttnRet
from loss import MSE_Loss
from preparers import Model_AttnRet_Preparer, Optimizer_Preparer, LoggerPreparer
import utils

class AttnRetTrainer:
    """AttnFactorVAE Trainer"""
    def __init__(self) -> None:
        
        self.model: AttnRet
        self.loss_func: MSE_Loss
        self.optimizer: torch.optim.optimizer.Optimizer
        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler

        self.train_loader:DataLoader
        self.val_loader:DataLoader

        self.model_preparer = Model_AttnRet_Preparer()
        self.optimizer_preparer = Optimizer_Preparer()
        self.dataloader_preparer = DataLoader_Preparer()

        self.logger:logging.Logger = None
        self.writer:SummaryWriter = None

        self.max_epoches:int
        self.grad_clip_value:float
        self.grad_clip_norm:float
        self.detect_anomaly:bool
        self.dtype:torch.dtype = torch.float32
        self.device:torch.device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.save_per_epoch:int = 1
        self.save_folder:str = os.curdir
        self.save_name:str = "Model"
        self.save_format:Literal[".pt", ".safetensors"] = ".pt"
        
        self.log_folder:str = "log"
        self.sample_per_batch:int = 0
        self.report_per_epoch:int = 1
        
    def save_checkpoint(self, 
                        save_folder:str, 
                        save_name:str, 
                        save_format:Literal[".pt",".safetensors"]=".pt"):
        utils.save_checkpoint(model=self.model, 
                              save_folder=save_folder, 
                              save_name=save_name, 
                              save_format=save_format)

    def set_logger(self, logger:logging.Logger):
        self.logger = logger
        self.model_preparer.set_logger(logger)
        self.dataloader_preparer.set_logger(logger)
        self.optimizer_preparer.set_logger(logger)

    def set_configs(self,
                    max_epoches:int,
                    grad_clip_value:float = None,
                    grad_clip_norm:float = None, 
                    detect_anomaly:bool = False, 
                    device:str = None,
                    dtype:str = None,
                    log_folder:str = "log",
                    sample_per_batch:int = 0,
                    report_per_epoch:int=1,
                    save_per_epoch:int=1,
                    save_folder:str=os.curdir,
                    save_name:str="Model",
                    save_format:str=".pt"):
        assert grad_clip_norm and grad_clip_value, "`grad_clip_norm` and `grad_clip_value` cannot be specified both"

        self.max_epoches = max_epoches
        self.grad_clip_value = grad_clip_value
        self.grad_clip_norm = grad_clip_norm
        self.detect_anomaly = detect_anomaly

        self.device = device
        self.dtype = dtype
        
        self.log_folder = log_folder
        self.sample_per_batch = sample_per_batch
        self.report_per_epoch = report_per_epoch
        self.save_per_epoch = save_per_epoch
        self.save_folder = save_folder
        self.save_name = save_name
        self.save_format = save_format

        os.makedirs(self.log_folder, exist_ok=True)
        os.makedirs(self.save_folder, exist_ok=True)

        self.writer = SummaryWriter(
            os.path.join(
                self.log_folder, f"TRAIN_{self.save_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            ))
        
    def load_configs(self, config_file:str):
        self.logger.info(f"Load hparams from config file `{config_file}`")
        train_configs = utils.read_configs(config_file=config_file)
        self.set_configs(max_epoches=train_configs["Train"]["max_epoches"],
                         grad_clip_norm=train_configs["Train"]["grad_clip_norm"],
                         grad_clip_value=train_configs["Train"]["grad_clip_value"],
                         detect_anomaly=train_configs["Train"]["detect_anomaly"], 
                         device=utils.str2device(train_configs["Train"]["device"]),
                         dtype=utils.str2dtype(train_configs["Train"]["dtype"]),
                         log_folder=train_configs["Train"]["log_folder"],
                         sample_per_batch=train_configs["Train"]["sample_per_batch"],
                         report_per_epoch=train_configs["Train"]["report_per_epoch"],
                         save_per_epoch=train_configs["Train"]["save_per_epoch"],
                         save_folder=train_configs["Train"]["save_folder"],
                         save_name=train_configs["Train"]["save_name"],
                         save_format=train_configs["Train"]["save_format"])

        self.model_preparer.load_configs(config_file=config_file)
        self.optimizer_preparer.load_configs(config_file=config_file)
        self.dataloader_preparer.load_configs(config_file=config_file)

    def load_args(self, args: argparse.Namespace | argparse.ArgumentParser):
        self.logger.info(f"Load hparams from argparser")
        if isinstance(args, argparse.ArgumentParser):
            args = args.parse_args()
        self.set_configs(max_epoches=args.max_epoches,
                         grad_clip_norm=args.grad_clip_norm,
                         grad_clip_value=args.grad_clip_value,
                         detect_anomaly=args.detect_anomaly, 
                         device=utils.str2device(args.device),
                         dtype=utils.str2dtype(args.dtype),
                         log_folder=args.log_folder,
                         sample_per_batch=args.sample_per_batch,
                         report_per_epoch=args.report_per_epoch,
                         save_per_epoch=args.save_per_epoch,
                         save_folder=args.save_folder,
                         save_name=args.save_name,
                         save_format=args.save_format) 
        
        self.model_preparer.load_args(args=args)
        self.optimizer_preparer.load_args(args=args)
        self.dataloader_preparer.load_args(args=args)

    def get_configs(self):
        train_configs = {"max_epoches": self.max_epoches, 
                         "grad_clip_norm": self.grad_clip_norm, 
                         "grad_clip_value": self.grad_clip_value,
                         "detect_anomaly": self.detect_anomaly, 
                         "device": self.device.type, 
                         "dtype": utils.dtype2str(self.dtype),
                         "log_folder": self.log_folder,
                         "sample_per_batch": self.sample_per_batch, 
                         "report_per_epoch": self.report_per_epoch, 
                         "save_per_epoch": self.save_per_epoch, 
                         "save_folder": self.save_folder, 
                         "save_name": self.save_name, 
                         "save_format": self.save_format}
        return train_configs
    
    def save_configs(self, config_file:Optional[str]=None):
        configs = {"Model": self.model_preparer.get_configs(),
                   "Optimizer": self.optimizer_preparer.get_configs(),
                   "Dataset": self.dataloader_preparer.get_configs(),
                   "Train": self.get_configs()}
        if not config_file:
            config_file = os.path.join(self.save_folder, "config.json")
        utils.save_configs(config_file=config_file, config_dict=configs)
        self.logger.info(f"Train Configs: {configs}`")
        self.logger.info(f"Config file saved to `{config_file}`")

    def prepare(self):
        self.model = self.model_preparer.prepare()
        self.loss_func = MSE_Loss()
        self.optimizer, self.lr_scheduler = self.optimizer_preparer.prepare(self.model.parameters())
        self.train_loader, self.val_loader, *_ = self.dataloader_preparer.prepare()
        
    def train(self):
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        torch.autograd.set_detect_anomaly(self.detect_anomaly)

        for epoch in range(self.max_epoches):
            train_loss_list = []
            val_loss_list = []
            
            self.model.train()
            for batch, (quantity_price_feature, fundamental_feature, label, valid_indices) in enumerate(tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{self.max_epoches}] Train")):    
                if fundamental_feature.shape[0] <= 2:
                    continue
                quantity_price_feature = quantity_price_feature.to(device=self.device)
                fundamental_feature = fundamental_feature.to(device=self.device)
                label = label.to(device=self.device)
                
                self.optimizer.zero_grad()

                y_hat = self.model(fundamental_feature, quantity_price_feature)
                train_loss = self.loss_func(label, y_hat)

                train_loss.backward()
                if self.grad_clip_value and self.grad_clip_value > 0:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.grad_clip_value)
                if self.grad_clip_norm and self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                self.optimizer.step()
                
                train_loss_list.append(train_loss.item())
                
                if self.sample_per_batch:
                    if (batch+1) % self.sample_per_batch == 0:
                        self.logger.debug(f"<Batch {batch+1}>  loss:{train_loss.item()}")
                        self.logger.debug(f"<Batch {batch+1}>  y_hat:{y_hat} label:{label}")
              
            train_loss_epoch = sum(train_loss_list)/len(train_loss_list)
            self.writer.add_scalar("Train Loss", train_loss_epoch, epoch+1)
            
            self.model.eval()
            with torch.no_grad(): 
                for batch, (quantity_price_feature, fundamental_feature, label, valid_indices) in enumerate(tqdm(self.val_loader, desc=f"Epoch [{epoch+1}/{self.max_epoches}] Val")):
                    if fundamental_feature.shape[0] <= 2:
                        continue
                    quantity_price_feature = quantity_price_feature.to(device=self.device)
                    fundamental_feature = fundamental_feature.to(device=self.device)
                    label = label.to(device=self.device)
                    
                    self.optimizer.zero_grad()

                    y_hat = self.model(fundamental_feature, quantity_price_feature)
                    val_loss = self.loss_func(label, y_hat)
                    val_loss_list.append(val_loss.item())

                val_loss_epoch = sum(val_loss_list) / len(val_loss_list)  
                self.writer.add_scalar("Validation Loss", val_loss_epoch, epoch+1)
                self.writer.add_scalars("Train-Val Loss", {"Train Loss": train_loss_epoch, "Validation Loss": val_loss_epoch}, epoch+1)

            self.writer.add_scalar("Learning Rate", self.optimizer.param_groups[0]["lr"], epoch+1)
            self.lr_scheduler.step()
            
            self.writer.flush()

            if self.report_per_epoch:
                if (epoch+1) % self.report_per_epoch == 0:
                    self.logger.info('Epoch [{}/{}], Train Loss: {:.6f}, Validation Loss: {:.6f}'.format(epoch+1, self.max_epoches, train_loss_epoch, val_loss_epoch))
            
            if self.save_per_epoch:
                if (epoch+1) % self.save_per_epoch == 0:
                    model_name = f"{self.save_name}_epoch{epoch+1}"
                    self.save_checkpoint(save_folder=self.save_folder,
                                         save_name=model_name,
                                         save_format=self.save_format)
                    self.logger.info(f"Epoch {epoch+1} Model weights saved to {os.path.join(self.save_folder)}")

        self.writer.close()

def get_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AttnRet Training.")

    parser.add_argument("--log_path", type=str, default="log/train_AttnRet.log", help="Path of log file. Default `log/train_AttnRet.log`")
    
    parser.add_argument("--load_configs", type=str, default=None, help="Path of config file to load. Optional")
    parser.add_argument("--save_configs", type=str, default=None, help="Path of config file to save. Default saved to save_folder as `config.json`")

    # dataloader config
    parser.add_argument("--dataset_path", type=str, help="Path of dataset .pt file")
    parser.add_argument("--num_workers", type=int, default=4, help="Num of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. Default 4")
    parser.add_argument("--shuffle", type=utils.str2bool, default=True, help="Whether to shuffle dataloader. Default True")
    parser.add_argument("--num_batches_per_epoch", type=int, default=-1, help="Num of batches sampled from all batches to be trained per epoch. Note that sampler option is mutually exclusive with shuffle. Specify -1 to disable (use all batches). Default -1")
    
    # model config
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path of checkpoint. Optional")
    parser.add_argument("--quantity_price_feature_size", type=int, help="Input size of quantity-price feature")
    parser.add_argument("--fundamental_feature_size", type=int, help="Input size of fundamental feature")
    parser.add_argument("--num_gru_layers", type=int, help="Num of GRU layers in feature extractor.")
    parser.add_argument("--gru_hidden_size", type=int, help="Hidden size of each GRU layer. num_gru_layers * gru_hidden_size i.e. the input size of FactorEncoder and Factor Predictor.")
    parser.add_argument("--gru_dropout", type=float, default=0.1, help="Dropout probs in gru layers. Default 0.1")
    parser.add_argument("--num_fc_layers", type=int, default=4, help="Num of full connected layers in MLP")

    # optimizer config
    parser.add_argument("--optimizer_type", type=str, default="Lion", choices=["Adam", "AdamW", "Lion", "SGDNesterov", "DAdaptation", "Adafactor"], help="Optimizer for training. Literally `Adam`, `AdamW`, `Lion`, `SGDNesterov`, `DAdaptation` or `Adafactor`. Default `Lion`")
    parser.add_argument("--optimizer_kwargs", type=str, default=None, nargs="+", help="Key arguments for optimizer. e.g. `betas=(0.9, 0.99) weight_decay=0.0 use_triton=False decoupled_weight_decay=False` for optimizer Lion by default")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer of AttnFactorVAE. Default 0.001")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant", choices=["constant", "linear", "cosine", "cosine_with_restarts", "polynomial", "adafactor"], help="Learning rate scheduler for optimizer. Literally `constant`, `linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `adafactor`. Default `constant`.")
    parser.add_argument("--lr_scheduler_warmup_steps", type=int, default=0, help="Number of steps for the warmup phase in the learning rate scheduler. Default 0")
    parser.add_argument("--lr_scheduler_num_cycles", type=float, default=0.5, help="Number of cycles (for cosine scheduler) or factor in polynomial scheduler. Default 0.5")
    parser.add_argument("--lr_scheduler_power", type=float, default=1.0, help="Power factor for polynomial learning rate scheduler. Default 1.0")

    # loss configs
    parser.add_argument("--gamma", type=float, default=1, help="Gamma for KL Div in Objective Function Loss. Default 1")
    parser.add_argument("--scale", type=float, default=100, help="Scale for MSE Loss. Default 100")
    
    # train configs
    parser.add_argument("--max_epoches", type=int, default=20, help="Max Epoches for train loop")
    parser.add_argument("--grad_clip_norm", type=float, default=-1, help="Value of gradient clipping. Specify -1 to disable. Default -1")
    parser.add_argument("--grad_clip_value", type=float, default=-1, help="Value of gradient clipping. Specify -1 to disable. Default -1")
    parser.add_argument("--detect_anomaly", type=utils.str2bool, default=False, help="Debug option. When enabled, PyTorch detects unusual operations (such as NaN or inf values) in the computation graph and throws exceptions to help locate the source of the problem. But it will greatly reduce the training performance. Default False")
    parser.add_argument("--dtype", type=str, default="FP32", choices=["FP32", "FP64", "FP16", "BF16"], help="Dtype of data and weight tensor. Literally `FP32`, `FP64`, `FP16` or `BF16`. Default `FP32`")
    parser.add_argument("--device", type=str, default="cuda", choices=["auto", "cuda", "cpu"], help="Device to take calculation. Literally `cpu` or `cuda`. Default `cuda`")
    parser.add_argument("--sample_per_batch", type=int, default=0, help="Check X, y and all kinds of outputs per n batches in one epoch. Specify 0 to disable. Default 0")
    parser.add_argument("--report_per_epoch", type=int, default=1, help="Report train loss and validation loss per n epoches. Specify 0 to disable. Default 1")
    parser.add_argument("--save_per_epoch", type=int, default=1, help="Save model weights per n epoches. Specify 0 to disable. Default 1")
    parser.add_argument("--save_folder", type=str, help="Folder to save model")
    parser.add_argument("--save_name", type=str, default="Model", help="Model name. Default `Model`")
    parser.add_argument("--save_format", type=str, default=".pt", help="File format of model to save, literally `.pt` or `.safetensors`. Default `.pt`")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_parser()
        
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    logger = LoggerPreparer(name="Train", 
                            file_level=logging.INFO, 
                            log_file=args.log_path).prepare()
    
    logger.debug(f"Command: {' '.join(sys.argv)}")

    trainer = AttnRetTrainer()
    trainer.set_logger(logger=logger)
    if args.load_configs:
        trainer.load_configs(config_file=args.load_configs)
    else:
        trainer.load_args(args=args)

    trainer.prepare()
    trainer.save_configs(config_file=args.save_configs)

    logger.info("Training start...")
    trainer.train()
    logger.info("Training complete.")
