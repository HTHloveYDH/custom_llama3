import os
import functools

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

from models.Transformer import Transformer as Llama
from models.DPOLlama import DPOLlama
from models.tensor_parallel import TP


def get_model(llama_config:dict, device, dp_strategy:str, dp_local_rank:int, device_mesh:dict):
    assert llama_config['load_weights'] in ['official', 'local', None], f"load weights: {llama_config['load_weights']}  is not supported"
    # create model
    if llama_config['load_weights'] == 'official':
        model = Llama.from_official_pretrained(llama_config)  # or init from Meta AI
    elif llama_config['load_weights'] == 'local':
        assert os.path.exists(llama_config['ckpt_path'])
        model = Llama.from_local_pretrained(llama_config)
    else:
        model = Llama.from_scratch(llama_config)
    if llama_config['align']:
        model = DPOLlama(model)
    if llama_config['lora']:
        model.init_lora(rank=llama_config['lora_rank'], alpha=llama_config['lora_alpha'])
    model.to(device)
    use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
    if use_compile:
        model = torch.compile(model)
    # tensor parallelism
    if device_mesh['tp'].size() > 1:
        tp_mesh = device_mesh['tp']
        dp_mesh = device_mesh['dp']  # update data parallel mesh
        model = TP(model, tp_mesh)
    else:
        dp_mesh = None
    # data parallelism
    if dp_strategy == 'ddp':
        model = DDP(model, device_ids=[dp_local_rank])
    elif dp_strategy == 'fsdp':
        # reference: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html#how-to-use-fsdp
        model = FSDP(model, device_mesh=dp_mesh, use_orig_params=True)
        # my_auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=100)
        # model = FSDP(
        #     model, auto_wrap_policy=my_auto_wrap_policy, cpu_offload=CPUOffload(offload_params=True), 
        #     device_mesh=dp_mesh, use_orig_params=True
        # )
    print(f'distribute strategy is set to {dp_strategy}')
    raw_model = model.module if dp_strategy in ['ddp', 'fsdp'] else model  # always contains the 'raw' unwrapped model
    return model, raw_model
