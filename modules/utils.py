import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal, Optional, List

def check(tensor:torch.Tensor):
    return torch.any(torch.isnan(tensor) | torch.isinf(tensor))

def multiLinear(input_size:int, 
                output_size:int, 
                num_layers:int=1, 
                nodes:Optional[List[int]]=None)->nn.Sequential:
    if nodes is None:
        if num_layers == 1:
            return nn.Linear(input_size, output_size)
        else:
            layers = []
            step = (input_size - output_size) // (num_layers - 1)
            for i in range(num_layers):
                in_features = input_size - i * step
                out_features = input_size - (i + 1) * step if i < num_layers - 1 else output_size
                layers.append(nn.Linear(in_features, out_features))
            return nn.Sequential(*layers)
    else:
        if len(nodes) == 1:
            return nn.Sequential(nn.Linear(input_size, nodes[0]),
                                 nn.Linear(nodes[0], output_size))
        else:
            layers = [nn.Linear(input_size, nodes[0])]
            for i in range(len(nodes)):
                layers.append(nn.Linear(nodes[i], nodes[i+1]))
            layers.append(nn.Linear(nodes[-1], output_size))
            return nn.Sequential(*layers)


class Exp(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def forward(self, x):
        return torch.exp(x)