'''reference url: 
1. https://github.com/pytorch/torchtitan/blob/main/torchtitan/utils.py
2. https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py#L274
'''
import os
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn

from models.ModelArgs import ModelArgs


def get_num_params(model:nn.Module, exclude_embedding:bool=False) -> int:
    num_params = sum(p.numel() for p in model.parameters())
    if exclude_embedding:
        num_params -= model.tok_embeddings.weight.numel()
    return num_params

def get_num_flop_per_token(num_params:int, model_args:ModelArgs) -> int:
    l, h, q, t = (
        model_args.n_layers,
        model_args.n_heads,
        model_args.dim // model_args.n_heads,
        model_args.max_seq_len,
    )
    # Reasoning behind the factor of 12 for the self-attention part of the formula:
    # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
    # 2. the flash attention does 1 more matmul recomputation in the backward
    #    but recomputation should not be counted in calculating MFU           (+0)
    # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
    # 4. we follow the convention and do not account for sparsity in causal attention
    flop_per_token = 6 * num_params + 12 * l * h * q * t
    return flop_per_token
  
