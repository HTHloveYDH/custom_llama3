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

from config.torch_config import TORCH_DTYPE_MAP
from dist.ParallelArgs import ParallelArgs
from dist.data_parallel import data_parallelize
from dist.tensor_parallel import tensor_parallelize
from dist.pipeline_parallel import pipeline_parallelize
from models.Transformer import Transformer as Llama
from models.DPOLlama import DPOLlama
from utils.model_utils import enable_activation_checkpoint, enable_compile


def get_model(llama_config:dict, parallel_args:ParallelArgs, device_mesh, device:str, \
              training:bool, **kwargs):
    assert llama_config['load_weights'] in ['official', 'local', None], f"load weights: {llama_config['load_weights']}  is not supported"
    # create model
    if llama_config['load_weights'] == 'official':
        model = Llama.from_official_pretrained(llama_config)  # or init from Meta AI
    elif llama_config['load_weights'] == 'local':
        assert os.path.exists(llama_config['ckpt_path'])
        model = Llama.from_local_pretrained(llama_config)
    else:
        model = Llama.from_scratch(llama_config)
    model.to(device)
    # dpo and optimizer
    optimizer = None
    if training:
        if llama_config['align']:
            model = DPOLlama(model)
        optimizer = _get_optimizer(model, **kwargs)
    # parallelism
    dp_mesh = None if parallel_args.dp == 1 else device_mesh['dp']
    tp_mesh = None if parallel_args.tp == 1 else device_mesh['tp']
    pp_mesh = None if parallel_args.pp == 1 else device_mesh['pp']
    # 2D parallel (tp + dp)
    if pp_mesh is None:
        pp_schedule = None
        if tp_mesh is not None:
            _ = tensor_parallelize(model, tp_mesh, training, parallel_args)
        if llama_config['activation_checkpoint_mode'] is not None:
            enable_activation_checkpoint(module, llama_config['activation_checkpoint_mode'])
        # turn on per-TransformerBlock compile after AC wrapping and before FSDP
        if llama_config['compile']:
            if model.params.norm_type == "fused_rmsnorm":
                raise NotImplementedError(
                    "fused_rmsnorm is not compatible with torch.compile yet. "
                    "Please use rmsnorm or layernorm."
                )
            enable_compile(model)
        # data parallelism
        if dp_mesh is not None:
            if model.dp_shard:
                # reference: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html#how-to-use-fsdp
                model = FSDP(model, device_mesh=dp_mesh, use_orig_params=True)
                # my_auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=100)
                # model = FSDP(
                #     model, auto_wrap_policy=my_auto_wrap_policy, cpu_offload=CPUOffload(offload_params=True),
                #     device_mesh=dp_mesh, use_orig_params=True
                # )
            else:
                model = DDP(model, device_ids=[device])
    # 3D parallel (pp + tp + dp)
    else:
        pp_schedule, modules = pipeline_parallelize(model, pp_mesh, training)
        # For PP with looped schedules, each item in model_parts is one stage-model-chunk.
        # We need to iterate through model_parts to apply SPMD parallelisms, compilation,
        # optimizer, and checkpointing
        for module in modules:
            # apply SPMD-style PT-D techniques
            if tp_mesh is not None:
                _ = tensor_parallelize(module, tp_mesh, training, parallel_args)
            if llama_config['activation_checkpoint_mode'] is not None:
                enable_activation_checkpoint(module, llama_config['activation_checkpoint_mode'])
            # turn on per-TransformerBlock compile after AC wrapping and before FSDP
            if llama_config['compile']:
                if model.params.norm_type == "fused_rmsnorm":
                    raise NotImplementedError(
                        "fused_rmsnorm is not compatible with torch.compile yet. "
                        "Please use rmsnorm or layernorm."
                    )
                enable_compile(module)
            if dp_mesh is not None:
                _ = data_parallelize(module, dp_mesh, training, parallel_args)
            module.train()
    return model, optimizer, pp_schedule

def _get_optimizer(raw_model, weight_decay:float, learning_rate:float):
    return torch.optim.Adam(raw_model.parameters())
