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

from dist.tensor_parallel import TP
from dist.pipeline_parallel import PP
from models.Transformer import Transformer as Llama
from models.DPOLlama import DPOLlama


def get_model(llama_config:dict, device_mesh:dict, device, training:bool, parallel_loss:bool, **kwargs):
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
    model.to(device)
    # optimizer
    optimizer = None
    if training:
        optimizer = _get_optimizer(model, **kwargs)
    use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
    if use_compile:
        model = torch.compile(model)
    # parallelism
    dp_mesh = None if llama_config['parallel_dims']['dp'] == 1 else device_mesh['dp']
    tp_mesh = None if llama_config['parallel_dims']['tp'] == 1 else device_mesh['tp']
    pp_mesh = None if llama_config['parallel_dims']['pp'] == 1 else device_mesh['pp']
    if tp_mesh is not None:
        model = TP(model, tp_mesh, training, parallel_loss)
    if pp_mesh is not None:
        model = PP(model, pp_mesh, training)
    # data parallelism
    if llama_config['parallel_dims']['dp'] > 1:
        if llama_config['dp_shard']:
            # reference: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html#how-to-use-fsdp
            model = FSDP(model, device_mesh=dp_mesh, use_orig_params=True)
            # my_auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=100)
            # model = FSDP(
            #     model, auto_wrap_policy=my_auto_wrap_policy, cpu_offload=CPUOffload(offload_params=True),
            #     device_mesh=dp_mesh, use_orig_params=True
            # )
        else:
            model = DDP(model, device_ids=[device])
    # print(f'distribute strategy is set to {dist_type}')
    return model, optimizer

def _get_optimizer(raw_model, weight_decay:float, learning_rate:float):
    return torch.optim.Adam(raw_model.parameters())
