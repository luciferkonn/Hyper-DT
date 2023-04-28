'''
Author: Jikun Kang
Date: 1969-12-31 19:00:00
LastEditTime: 2023-04-28 17:12:42
LastEditors: Jikun Kang
FilePath: /Hyper-DT/src/lora_utils.py
'''
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from functools import partial
from relational_memory import RelationalMemory

default_lora_config = {
    nn.Linear: {
        "weight": partial(RelationalMemory.from_linear, n_head=1)
    }
}


def apply_lora(
    layer,
    register=True,
    merge=False,
    lora_config=default_lora_config,
):
    if register:
        if type(layer) in lora_config:
            for attr_name, parameterization in lora_config[type(layer).items()]:
                parametrize.register_parametrization(
                    layer, attr_name, parameterization(layer))
    else:
        if hasattr(layer, "parameterizations"):
            for attr_name in layer.parameterizations.keys():
                parametrize.remove_parametrizations(
                    layer, attr_name, leave_parametrized=merge)


def add_lora(
        model,
        lora_config=default_lora_config,
):
    model.apply(partial(apply_lora, lora_config=lora_config))


def merge_lora(model):
    model.apply(partial(apply_lora, register=False, merge=True))


def remove_lora(model):
    model.apply(partial(apply_lora, register=False, merge=False))
