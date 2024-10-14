import os
import ast
import argparse
from typing import Callable, Literal, Union, Optional, List, Dict, Any
from collections import OrderedDict
import logging
from logging import Logger
from logging.handlers import RotatingFileHandler

import safetensors.torch
import torch
import lion_pytorch
import dadaptation
import diffusers
import transformers
import safetensors

from nets import *
from loss import *
import utils

class Model_WeightTransfer:
    def __init__(self):
        self.logger:Logger
        self.state_dict:OrderedDict

    def set_logger(self, logger:Logger):
        if not logger:
            logger = logging
        self.logger = logger

    def load_state_dict(self, path:str):
        if path.endswith(".pt"):
            self.state_dict = torch.load(path, weights_only=False)
        elif path.endswith(".safetensors"):
            self.state_dict = safetensors.torch.load_file(path)
        else:
            raise ValueError("Unrecognized checkpoint file")

    def afvae2fvae(self, dest:FactorVAE):
        encoder_state_dict = {k.removeprefix("encoder."): v for k, v in self.state_dict.items() if k.startswith("encoder.")}
        decoder_state_dict = {k.removeprefix("decoder."): v for k, v in self.state_dict.items() if k.startswith("decoder.")}
        predictor_state_dict = {k.removeprefix("predictor."): v for k, v in self.state_dict.items() if k.startswith("predictor.")}
        dest.encoder.load_state_dict(state_dict=encoder_state_dict)
        dest.decoder.load_state_dict(state_dict=decoder_state_dict)
        dest.predictor.load_state_dict(state_dict=predictor_state_dict)

    def afvae2riskattn(self, dest:AttnRet):
        feature_extractor_state_dict = {k.removeprefix("feature_extractor."): v for k, v in self.state_dict.items() if k.startswith("feature_extractor.")}
        dest.feature_extractor.load_state_dict(state_dict=feature_extractor_state_dict)

    def fvae2afvae(self, dest:AttnFactorVAE):
        encoder_state_dict = {k.removeprefix("encoder."): v for k, v in self.state_dict.items() if k.startswith("encoder.")}
        decoder_state_dict = {k.removeprefix("decoder."): v for k, v in self.state_dict.items() if k.startswith("decoder.")}
        predictor_state_dict = {k.removeprefix("predictor."): v for k, v in self.state_dict.items() if k.startswith("predictor.")}
        dest.encoder.load_state_dict(state_dict=encoder_state_dict)
        dest.decoder.load_state_dict(state_dict=decoder_state_dict)
        dest.predictor.load_state_dict(state_dict=predictor_state_dict)

    def riskattn2afvae(self, dest:AttnFactorVAE):
        feature_extractor_state_dict = {k.removeprefix("feature_extractor."): v for k, v in self.state_dict.items() if k.startswith("feature_extractor.")}
        dest.feature_extractor.load_state_dict(state_dict=feature_extractor_state_dict)

class Preparer:
    def __init__(self):
        self.logger:Logger = None
        self.configs:Dict[str, Any] = {}

    def set_configs(self):
        ...

    def load_configs(self, config_file:str):
        configs = utils.read_configs(config_file=config_file)
        return configs

    def load_args(self, args:Union[argparse.Namespace, argparse.ArgumentParser]):
        if isinstance(args, argparse.ArgumentParser):
            args = args.parse_args()
        return args

    def get_args(self):
        return self.configs
    
    def set_logger(self, logger:Logger):
        if not logger:
            logger = logging
        self.logger = logger

    def prepare(self):
        ...


class Model_Preparer(Preparer):
    def __init__(self, model_type) -> None:
        super().__init__()
        self.model_type:str = model_type
        self.checkpoint_path:str = None

    def load_configs(self, config_file:str) -> Dict[str, Any]:
        configs = super().load_configs(config_file=config_file)
        if configs["Model"]["type"] == self.model_type:
            return configs
        else:
            raise TypeError(f"Wrong config model type `{configs["Model"]["type"]}` mismatch `{self.model_type}`")
    

class Model_AttnFactorVAE_Preparer(Model_Preparer):
    def __init__(self) -> None:
        super(Model_AttnFactorVAE_Preparer, self).__init__(model_type="AttnFactorVAE")

        self.fundamental_feature_size:int
        self.quantity_price_feature_size:int
        self.num_gru_layers:int
        self.gru_hidden_size:int
        self.hidden_size:int
        self.latent_size:int
        self.gru_dropout:float = 0.1
        self.std_activation:Literal["exp", "softplus"] = "exp"
    
    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if hasattr(self, "configs") and name != "configs" and name != "logger":
            self.configs[name] = value

    def set_configs(self, 
                    fundamental_feature_size,
                    quantity_price_feature_size,
                    num_gru_layers,
                    gru_hidden_size,
                    hidden_size,
                    latent_size,
                    gru_dropout,
                    std_activation,
                    checkpoint_path=None):
        self.fundamental_feature_size = fundamental_feature_size
        self.quantity_price_feature_size = quantity_price_feature_size
        self.num_gru_layers = num_gru_layers
        self.gru_hidden_size = gru_hidden_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.gru_dropout = gru_dropout
        self.std_activation = std_activation
        self.checkpoint_path = checkpoint_path

    def load_configs(self, config_file: str) -> Dict[str, Any]:
        configs = super().load_configs(config_file)
        self.set_configs(fundamental_feature_size=configs["Model"]["fundamental_feature_size"],
                         quantity_price_feature_size=configs["Model"]["quantity_price_feature_size"],
                         num_gru_layers=configs["Model"]["num_gru_layers"],
                         gru_dropout=configs["Model"]["gru_dropout"],
                         gru_hidden_size=configs["Model"]["gru_hidden_size"],
                         hidden_size=configs["Model"]["hidden_size"],
                         latent_size=configs["Model"]["latent_size"],
                         std_activation=configs["Model"]["std_activation"],
                         checkpoint_path=configs["Model"]["checkpoint_path"])
    
    def get_configs(self):
        return self.configs
        
    def load_args(self, args: argparse.Namespace | argparse.ArgumentParser):
        args = super().load_args(args)
        self.set_configs(fundamental_feature_size=args.fundamental_feature_size,
                         quantity_price_feature_size=args.quantity_price_feature_size,
                         num_gru_layers=args.num_gru_layers,
                         gru_dropout=args.gru_dropout,
                         gru_hidden_size=args.gru_hidden_size,
                         hidden_size=args.hidden_size,
                         latent_size=args.latent_size,
                         std_activation=args.std_activation,
                         checkpoint_path=args.checkpoint_path)
        
    def prepare(self):
        model = AttnFactorVAE(fundamental_feature_size=self.fundamental_feature_size,
                              quantity_price_feature_size=self.quantity_price_feature_size,
                              num_gru_layers=self.num_gru_layers,
                              gru_hidden_size=self.gru_hidden_size,
                              hidden_size=self.hidden_size,
                              latent_size=self.latent_size,
                              gru_drop_out=self.gru_dropout,
                              std_activ=self.std_activation)
        if self.checkpoint_path:
            self.load_checkpoint(model=model, checkpoint_path=self.checkpoint_path)
        return model

    def load_checkpoint(self, model:FactorVAE, checkpoint_path:str):
        checkpoint_folder = os.path.dirname(checkpoint_path)
        if not checkpoint_folder:
            checkpoint_folder = os.curdir
        if os.path.exists(os.path.join(checkpoint_folder, "config.json")):
            checkpoint_config = utils.read_configs(os.path.join(checkpoint_folder, "config.json"))
            model_type = checkpoint_config["Model"]["type"]
            if model_type == "AttnFactorVAE":
                self.logger.info("Loading AttnFactorVAE weights...")
                utils.load_checkpoint(model, checkpoint_path)
            elif model_type == "FactorVAE":
                self.logger.info("Loading encoder, decoder, and predictor weights from FactorVAE model...")
                utils.check_attr_dict_match(self, checkpoint_config["Model"], names=["hidden_size", "latent_size"])
                transfer = Model_WeightTransfer()
                transfer.set_logger(logger=self.logger)
                transfer.load_state_dict(checkpoint_path)
                transfer.fvae2afvae(dest=model)
            elif model_type == "RiskAttention":
                self.logger.info("Loading attn_feature_extractor weights from the RiskAttention model...")
                utils.check_attr_dict_match(self, 
                                            checkpoint_config["Model"], 
                                            names=["fundamental_feature_size", 
                                                   "quantity_price_feature_size",
                                                   "num_gru_layers",
                                                   "gru_hidden_size",
                                                   "gru_drop_out"])
                transfer = Model_WeightTransfer()
                transfer.set_logger(logger=self.logger)
                transfer.load_state_dict(checkpoint_path)
                transfer.riskattn2afvae(dest=model)
            else:
                self.logger.warning(f"Model type `{model_type}` detected that does not support migration. Default randomly initialized weights will be used.")
        else:
            self.logger.critical(f"No configuration files found in the checkpoint folder `{checkpoint_folder}`. Try loading weights...")
            try:
                utils.load_checkpoint(model, checkpoint_path)
                self.logger.info("Loading weight successfully.")
            except Exception as e:
                self.logger.warning("Failed to load weight. Default randomly initialized weights will be used.")
    

class Model_FactorVAE_Preparer(Model_Preparer):
    def __init__(self) -> None:
        super(Model_FactorVAE_Preparer, self).__init__(model_type="FactorVAE")

        self.feature_size:int
        self.num_gru_layers:int
        self.gru_hidden_size:int
        self.hidden_size:int
        self.latent_size:int
        self.gru_dropout:float = 0.1
        self.std_activation:Literal["exp", "softplus"] = "softplus"
    
    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if hasattr(self, "configs") and name != "configs" and name != "logger":
            self.configs[name] = value

    def set_configs(self, 
                    feature_size,
                    num_gru_layers,
                    gru_hidden_size,
                    hidden_size,
                    latent_size,
                    gru_dropout,
                    std_activation,
                    checkpoint_path=None):
        self.feature_size = feature_size
        self.num_gru_layers = num_gru_layers
        self.gru_hidden_size = gru_hidden_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.gru_dropout = gru_dropout
        self.std_activation = std_activation
        self.checkpoint_path = checkpoint_path

    def load_configs(self, config_file: str) -> Dict[str, Any]:
        configs = super().load_configs(config_file)
        self.set_configs(feature_size=configs["Model"]["feature_size"],
                         num_gru_layers=configs["Model"]["num_gru_layers"],
                         gru_dropout=configs["Model"]["gru_dropout"],
                         gru_hidden_size=configs["Model"]["gru_hidden_size"],
                         hidden_size=configs["Model"]["hidden_size"],
                         latent_size=configs["Model"]["latent_size"],
                         std_activation=configs["Model"]["std_activation"],
                         checkpoint_path=configs["Model"]["checkpoint_path"])
    
    def get_configs(self):
        return self.configs
        
    def load_args(self, args: argparse.Namespace | argparse.ArgumentParser):
        args = super().load_args(args)
        self.set_configs(feature_size=args.feature_size,
                         num_gru_layers=args.num_gru_layers,
                         gru_dropout=args.gru_dropout,
                         gru_hidden_size=args.gru_hidden_size,
                         hidden_size=args.hidden_size,
                         latent_size=args.latent_size,
                         std_activation=args.std_activation,
                         checkpoint_path=args.checkpoint_path)
        
    def prepare(self):
        model = FactorVAE(feature_size=self.feature_size,
                          num_gru_layers=self.num_gru_layers,
                          gru_hidden_size=self.gru_hidden_size,
                          hidden_size=self.hidden_size,
                          latent_size=self.latent_size,
                          gru_drop_out=self.gru_dropout,
                          std_activ=self.std_activation)
        if self.checkpoint_path:
            self.load_checkpoint(model=model, checkpoint_path=self.checkpoint_path)
        return model
    
    def load_checkpoint(self, model:FactorVAE, checkpoint_path:str):
        checkpoint_folder = os.path.dirname(checkpoint_path)
        if not checkpoint_folder:
            checkpoint_folder = os.curdir
        if os.path.exists(os.path.join(checkpoint_folder, "config.json")):
            checkpoint_config = utils.read_configs(os.path.join(checkpoint_folder, "config.json"))
            model_type = checkpoint_config["Model"]["type"]
            if model_type == "FactorVAE":
                self.logger.info("Loading FactorVAE weights...")
                utils.load_checkpoint(model, checkpoint_path)
            elif model_type == "AttnFactorVAE":
                self.logger.info("Loading encoder, decoder, and predictor weights from the AttnFactorVAE model...")
                utils.check_attr_dict_match(self, checkpoint_config["Model"], names=["hidden_size", "latent_size"])
                transfer = Model_WeightTransfer()
                transfer.set_logger(logger=self.logger)
                transfer.load_state_dict(checkpoint_path)
                transfer.afvae2fvae(dest=model)
            else:
                self.logger.warning(f"Model type `{model_type}` detected that does not support migration. Default randomly initialized weights will be used.")
        else:
            self.logger.critical(f"No configuration files found in the checkpoint folder `{checkpoint_folder}`. Try loading weights...")
            try:
                utils.load_checkpoint(model, checkpoint_path)
                self.logger.info("Loading weight successfully.")
            except Exception as e:
                self.logger.warning("Failed to load weight. Default randomly initialized weights will be used.")


class Model_RiskAttention_Preparer(Model_Preparer):
    def __init__(self) -> None:
        super(Model_FactorVAE_Preparer, self).__init__(model_type="RiskAttention")

        self.fundamental_feature_size:int
        self.quantity_price_feature_size:int
        self.num_gru_layers:int
        self.gru_hidden_size:int
        self.gru_dropout:float = 0.1
        self.num_fc_layers:int
    
    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if hasattr(self, "configs") and name != "configs" and name != "logger":
            self.configs[name] = value

    def set_configs(self, 
                    fundamental_feature_size,
                    quantity_price_feature_size,
                    num_gru_layers,
                    gru_hidden_size,
                    gru_dropout,
                    num_fc_layers,
                    checkpoint_path=None):
        self.fundamental_feature_size = fundamental_feature_size
        self.quantity_price_feature_size = quantity_price_feature_size
        self.num_gru_layers = num_gru_layers
        self.gru_hidden_size = gru_hidden_size
        self.gru_dropout = gru_dropout
        self.num_fc_layers = num_fc_layers
        self.checkpoint_path = checkpoint_path

    def load_configs(self, config_file: str) -> Dict[str, Any]:
        configs = super().load_configs(config_file)
        self.set_configs(fundamental_feature_size=configs["Model"]["fundamental_feature_size"],
                         quantity_price_feature_size=configs["Model"]["quantity_price_feature_size"],
                         num_gru_layers=configs["Model"]["num_gru_layers"],
                         gru_dropout=configs["Model"]["gru_dropout"],
                         gru_hidden_size=configs["Model"]["gru_hidden_size"],
                         num_fc_layers=configs["Model"]["num_fc_layers"],
                         checkpoint_path=configs["Model"]["checkpoint_path"])
    
    def get_configs(self):
        return self.configs
        
    def load_args(self, args: argparse.Namespace | argparse.ArgumentParser):
        args = super().load_args(args)
        self.set_configs(fundamental_feature_size=args.fundamental_feature_size,
                         quantity_price_feature_size=args.quantity_price_feature_size,
                         num_gru_layers=args.num_gru_layers,
                         gru_dropout=args.gru_dropout,
                         gru_hidden_size=args.gru_hidden_size,
                         num_fc_layers=args.num_fc_layers,
                         checkpoint_path=args.checkpoint_path)
        
    def prepare(self):
        model = AttnRet(fundamental_feature_size=self.fundamental_feature_size,
                        quantity_price_feature_size=self.quantity_price_feature_size,
                        num_gru_layers=self.num_gru_layers,
                        gru_hidden_size=self.gru_hidden_size,
                        gru_drop_out=self.gru_dropout,
                        num_fc_layers=self.num_fc_layers)
        if self.checkpoint_path:
            self.load_checkpoint(model=model, checkpoint_path=self.checkpoint_path)
        return model
    
    def load_checkpoint(self, model:FactorVAE, checkpoint_path:str):
        checkpoint_folder = os.path.dirname(checkpoint_path)
        if not checkpoint_folder:
            checkpoint_folder = os.curdir
        if os.path.exists(os.path.join(checkpoint_folder, "config.json")):
            checkpoint_config = utils.read_configs(os.path.join(checkpoint_folder, "config.json"))
            model_type = checkpoint_config["Model"]["type"]
            if model_type == "RiskAttention":
                self.logger.info("Loading RiskAttention weights...")
                utils.load_checkpoint(model, checkpoint_path)
            elif model_type == "AttnFactorVAE":
                self.logger.info("Loading attn_feature_extractor weights from the AttnFactorVAE model...")
                utils.check_attr_dict_match(self, 
                                            checkpoint_config["Model"], 
                                            names=["fundamental_feature_size", 
                                                   "quantity_price_feature_size",
                                                   "num_gru_layers",
                                                   "gru_hidden_size",
                                                   "gru_drop_out"])
                transfer = Model_WeightTransfer()
                transfer.set_logger(logger=self.logger)
                transfer.load_state_dict(checkpoint_path)
                transfer.afvae2riskattn(dest=model)
            else:
                self.logger.warning(f"Model type `{model_type}` detected that does not support migration. Default randomly initialized weights will be used.")
        else:
            self.logger.critical(f"No configuration files found in the checkpoint folder `{checkpoint_folder}`. Try loading weights...")
            try:
                utils.load_checkpoint(model, checkpoint_path)
                self.logger.info("Loading weight successfully.")
            except Exception as e:
                self.logger.warning("Failed to load weight. Default randomly initialized weights will be used.")


class ObjectiveLoss_Preparer(Preparer):
    def __init__(self) -> None:
        super().__init__()
        self.gamma:float = 1.0
        self.scale:float = 100.0

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if hasattr(self, "configs") and name != "configs" and name != "logger":
            self.configs[name] = value
    
    def set_configs(self, gamma, scale):
        self.gamma = gamma
        self.scale = scale
    
    def load_configs(self, config_file:str):
        configs = super().load_configs(config_file)
        self.set_configs(gamma=configs["Objective_Loss"]["gamma"], 
                         scale=configs["Objective_Loss"]["scale"])
    
    def load_args(self, args: argparse.Namespace | argparse.ArgumentParser):
        args = super().load_args(args)
        self.set_configs(gamma=args.gamma, scale=args.scale)

    def get_configs(self):
        return self.configs
    
    def prepare(self):
        return ObjectiveLoss(scale=self.scale, gamma=self.gamma)

class Optimizer_Preparer(Preparer):
    def __init__(self, optimizer_name:Optional[str] = None) -> None:
        super().__init__()

        self.optimizer_name = optimizer_name or "Optimizer"
        self.optimizer_type:str
        self.optimizer_kwargs:Dict
        self.learning_rate:float
        self.lr_scheduler_type:Literal["Adam", "AdamW", "Lion", "SGDNesterov", "DAdaptation", "Adafactor"]
        self.lr_scheduler_warmup_steps:int
        self.lr_scheduler_train_steps:int
        self.lr_scheduler_num_cycles:int
        self.lr_scheduler_power:int

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if hasattr(self, "configs") and name != "configs" and name != "logger" and name != "lr_scheduler_train_steps":
            self.configs[name] = value
    
    def set_configs(self, 
                    optimizer_type,
                    optimizer_kwargs, 
                    learning_rate, 
                    lr_scheduler_type, 
                    lr_scheduler_warmup_steps, 
                    lr_scheduler_train_steps,
                    lr_scheduler_num_cycles, 
                    lr_scheduler_power):
        self.optimizer_type = optimizer_type
        self.optimizer_kwargs = optimizer_kwargs
        self.learning_rate = learning_rate
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_scheduler_warmup_steps = lr_scheduler_warmup_steps
        self.lr_scheduler_train_steps = lr_scheduler_train_steps
        self.lr_scheduler_num_cycles = lr_scheduler_num_cycles
        self.lr_scheduler_power = lr_scheduler_power

    def load_configs(self, config_file:str):
        configs = super().load_configs(config_file)
        self.set_configs(optimizer_type=configs[self.optimizer_name]["optimizer_type"], 
                         optimizer_kwargs=configs[self.optimizer_name]["optimizer_kwargs"], 
                         learning_rate=configs[self.optimizer_name]["learning_rate"], 
                         lr_scheduler_type=configs[self.optimizer_name]["lr_scheduler_type"], 
                         lr_scheduler_warmup_steps=configs[self.optimizer_name]["lr_scheduler_warmup_steps"],
                         lr_scheduler_train_steps=configs["Train"]["max_epoches"],  
                         lr_scheduler_num_cycles=configs[self.optimizer_name]["lr_scheduler_num_cycles"], 
                         lr_scheduler_power=configs[self.optimizer_name]["lr_scheduler_power"])
        
    def get_configs(self):
        return self.configs
    
    def load_args(self, args: argparse.Namespace | argparse.ArgumentParser):
        args = super().load_args(args)
        name = self.optimizer_name.lower().removesuffix("_optimizer")
        self.set_configs(optimizer_type=getattr(args, f"{name}_optimizer_type"), 
                         optimizer_kwargs=utils.str2dict(getattr(args, f"{name}_optimizer_kwargs")), 
                         learning_rate=getattr(args, f"{name}_learning_rate"), 
                         lr_scheduler_type=getattr(args, f"{name}_lr_scheduler_type"), 
                         lr_scheduler_warmup_steps=getattr(args, f"{name}_lr_scheduler_warmup_steps"), 
                         lr_scheduler_train_steps=args.max_epoches, 
                         lr_scheduler_num_cycles=getattr(args, f"{name}_lr_scheduler_num_cycles"), 
                         lr_scheduler_power=getattr(args, f"{name}_lr_scheduler_power"))
    
    def prepare_optimizer(self, trainable_params) -> torch.optim.Optimizer:
        if self.optimizer_type == "AdamW":
            optimizer_class = torch.optim.AdamW
            optimizer = optimizer_class(trainable_params, lr=self.learning_rate, **self.optimizer_kwargs)
        elif self.optimizer_type == "Adam":
            optimizer_class = torch.optim.Adam
            optimizer = optimizer_class(trainable_params, lr=self.learning_rate, **self.optimizer_kwargs)
        elif self.optimizer_type == "Lion":
            optimizer_class = lion_pytorch.Lion
            optimizer = optimizer_class(trainable_params, 
                                        lr=self.learning_rate, 
                                        **self.optimizer_kwargs)
        elif self.optimizer_type == "SGDNesterov":
            if "momentum" not in self.optimizer_kwargs:
                self.logger.critical(f"SGD with Nesterov must be with momentum, set momentum to 0.9")
                self.optimizer_kwargs["momentum"] = 0.9
            optimizer_class = torch.optim.SGD
            optimizer = optimizer_class(trainable_params, 
                                        lr=self.learning_rate, 
                                        nesterov=True, 
                                        **self.optimizer_kwargs)
        elif self.optimizer_type == "DAdaptation":
            optimizer_class = dadaptation.DAdaptAdam
            optimizer = optimizer_class(trainable_params, 
                                        lr=self.learning_rate, 
                                        **self.optimizer_kwargs)
        elif self.optimizer_type == "Adafactor":
            if "relative_step" not in self.optimizer_kwargs:
                self.optimizer_kwargs["relative_step"] = True  # default
            if not self.optimizer_kwargs["relative_step"] and self.optimizer_kwargs.get("warmup_init", False):
                self.logger.info(f"set relative_step to True because warmup_init is True.")
                self.optimizer_kwargs["relative_step"] = True
            if self.optimizer_kwargs["relative_step"]:
                if self.learning_rate != 0.0:
                    self.logger.info(f"Learning rate is used as initial_lr.")
                if self.lr_scheduler_type != "adafactor":
                    self.logger.info(f"Use adafactor_scheduler.")
                    self.lr_scheduler_type = "adafactor"
            optimizer_class = transformers.optimization.Adafactor
            optimizer = optimizer_class(trainable_params, lr=None, **self.optimizer_kwargs)
        return optimizer
    
    def prepare_lr_scheduler(self, optimizer:torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        if self.lr_scheduler_type.lower() == "constant":
            lr_scheduler = diffusers.optimization.get_constant_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=self.lr_scheduler_warmup_steps)
        elif self.lr_scheduler_type.lower() == "linear":
            lr_scheduler = diffusers.optimization.get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=self.lr_scheduler_warmup_steps, 
                num_training_steps=self.lr_scheduler_train_steps)
        elif self.lr_scheduler_type.lower() == "cosine":
            lr_scheduler = diffusers.optimization.get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.lr_scheduler_warmup_steps,
                num_training_steps=self.lr_scheduler_train_steps, 
                num_cycles=self.lr_scheduler_num_cycles)
        elif self.lr_scheduler_type.lower() == "cosine_with_restarts":
            lr_scheduler = diffusers.optimization.get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=self.lr_scheduler_warmup_steps, 
                num_training_steps=self.lr_scheduler_train_steps,
                num_cycles=self.lr_scheduler_num_cycles)
        elif self.lr_scheduler_type.lower() == "polynomial":
            lr_scheduler = diffusers.optimization.get_polynomial_decay_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=self.lr_scheduler_warmup_steps,
                num_training_steps=self.lr_scheduler_train_steps,
                power=self.lr_scheduler_power)
        elif self.lr_scheduler_type.lower() == "adafactor":
            assert type(optimizer) == transformers.optimization.Adafactor, f"Adafactor Scheduler must be used with Adafactor Optimizer. Unexpected optimizer type {type(optimizer)}"
            lr_scheduler = transformers.optimization.AdafactorSchedule(optimizer, initial_lr=self.learning_rate)
        return lr_scheduler
    
    def prepare(self, trainable_params):
        optimizer = self.prepare_optimizer(trainable_params=trainable_params)
        lr_scheduler = self.prepare_lr_scheduler(optimizer=optimizer)
        return optimizer, lr_scheduler
    
class LoggerPreparer:
    def __init__(self, 
                 name: str = 'app_logger', 
                 console_level=logging.DEBUG, 
                 file_level=logging.DEBUG,
                 log_file: str = 'app.log', 
                 max_bytes: int = 1e6, 
                 backup_count: int = 5):

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)

        file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        file_handler.setLevel(file_level)

        formatter = logging.Formatter('%(asctime)s [%(name)s][%(levelname)s]: %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def get_logger(self) -> Logger:
        return self.logger