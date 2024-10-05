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

def convert(ckpt_path:str, format:str, save_dir:str, splits=3):
    import json

    from models.Transformer import Transformer
    
    def _convert_to_onnx(model:torch.nn.Module, save_dir:str):
        model.train()
        dummy_input = torch.randn(1, model.params.max_seq_len, dtype=torch.long)
        torch.onnx._export(
            model,
            dummy_input,
            os.path.join(save_dir, 'llama.onnx'),
            input_names=['input'],
            output_names=['output'],
            opset_version=13
        )

    def _convert_to_safetensors(model:torch.nn.Module, save_dir:str, splits:int):
        from safetensors.torch import save_file
        state_dict = model.state_dict()
        keys = list(state_dict.keys())
        T = len(keys) // splits
        for i in range(splits - 1):
            state_dict_split = OrderedDict()
            for j in range(i * T, (i + 1) * T):
                state_dict_split[keys[j]] = state_dict[keys[j]]
            save_file(state_dict_split, os.path.join(save_dir, f'./model_split_{i}.safetensors'))

    with open('./config/llama_config.json', 'r') as f:
        llama_config = json.load(f)
    ckpt = torch.load(ckpt_path)
    if hasattr(ckpt, 'model'):
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    llama = Transformer.from_local_pretrained(llama_config)
    llama.load_state_dict(state_dict)
    if format == 'onnx':
        _convert_to_onnx(llama)
    elif format == 'safetensors':
        _convert_to_safetensors(llama, splits)
