import os

import torch

from models.Transformer import Transformer as Llama
from models.DPOLlama import DPOLlama
from models.utils import get_num_params, get_num_flop_per_token


def get_model(llama_config:dict, device:str, training:bool, **kwargs):
    assert llama_config['load_weights'] in ['official', 'local', None], f"load weights: {llama_config['load_weights']}  is not supported"
    # create model
    if llama_config['load_weights'] == 'official':
        model = Llama.from_official_pretrained(llama_config)  # or init from Meta AI
    elif llama_config['load_weights'] == 'local':
        assert os.path.exists(llama_config['ckpt_path'])
        model = Llama.from_local_pretrained(llama_config)
    else:
        model = Llama.from_scratch(llama_config)
    # log model size
    model_param_count = get_num_params(model)
    num_flop_per_token = get_num_flop_per_token(
        get_num_params(model, exclude_embedding=True), 
        model.params
    )
    print(f'model param count: {model_param_count}')
    print(f'flops per token: {num_flop_per_token}')
    model.to(device)
    # dpo and optimizer
    optimizer = None
    if training:
        if llama_config['align']:
            model = DPOLlama(model)
        optimizer = _get_optimizer(model, **kwargs)
    return model, optimizer

def _get_optimizer(raw_model, weight_decay:float, learning_rate:float):
    return torch.optim.Adam(raw_model.parameters())
