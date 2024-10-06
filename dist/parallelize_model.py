import torch
import torch.nn as nn
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
from dist.data_parallel import enable_data_parallel
from dist.tensor_parallel import enable_tensor_parallel
from dist.pipeline_parallel import enable_pipeline_parallel
from dist.utils import enable_activation_checkpoint, enable_compile


def parallelize_model(model:nn.Module, parallel_args:ParallelArgs, device_mesh, training:bool):
    # parallelism
    dp_mesh = None if parallel_args.dp == 1 else device_mesh['dp']
    tp_mesh = None if parallel_args.tp == 1 else device_mesh['tp']
    pp_mesh = None if parallel_args.pp == 1 else device_mesh['pp']
    # 2D parallel (tp + dp)
    if pp_mesh is None:
        modules = [model]
        pp_schedule = None
        if tp_mesh is not None:
            assert not (parallel_args.async_tp and not parallel_args.compile), ‘Async TP requires --parallel_args.compile’
            enable_tensor_parallel(model, tp_mesh, training, parallel_args)
        if parallel_args.activation_checkpoint_mode is not None:
            enable_activation_checkpoint(module, parallel_args.activation_checkpoint_mode)
        # turn on per-TransformerBlock compile after AC wrapping and before FSDP
        if parallel_args.compile:
            assert not model.params.norm_type == 'fused_rmsnorm', 'fused_rmsnorm is not compatible with torch.compile yet. Please use rmsnorm or layernorm.'
            enable_compile(model)
        # data parallelism
        if dp_mesh is not None:
            if parallel_args.dp_shard:
                # reference: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html#how-to-use-fsdp
                model = FSDP(model, device_mesh=dp_mesh, use_orig_params=True)
                # my_auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=100)
                # model = FSDP(
                #     model, auto_wrap_policy=my_auto_wrap_policy, cpu_offload=CPUOffload(offload_params=True),
                #     device_mesh=dp_mesh, use_orig_params=True
                # )
            else:
                assert tp_mesh is None and pp_mesh is None, 'DDP has not supported > 1D parallelism'
                model = DDP(model, device_ids=[device])
    # 3D parallel (pp + tp + dp)
    else:
        pp_schedule, modules = enable_pipeline_parallel(
            model, pp_mesh, training, parallel_args, model.params
        )
        # For PP with looped schedules, each item in model_parts is one stage-model-chunk.
        # We need to iterate through model_parts to apply SPMD parallelisms, compilation,
        # optimizer, and checkpointing
        for module in modules:
            # apply SPMD-style PT-D techniques
            if tp_mesh is not None:
                assert not (parallel_args.async_tp and not parallel_args.compile), ‘Async TP requires --parallel_args.compile’
                enable_tensor_parallel(module, tp_mesh, training, parallel_args)
            if parallel_args.activation_checkpoint_mode is not None:
                enable_activation_checkpoint(module, parallel_args.activation_checkpoint_mode)
            # turn on per-TransformerBlock compile after AC wrapping and before FSDP
            if papallel_args.compile:
                assert not model.params.norm_type == 'fused_rmsnorm', 'fused_rmsnorm is not compatible with torch.compile yet. Please use rmsnorm or layernorm.'
                enable_compile(module)
            if dp_mesh is not None:
                enable_data_parallel(module, dp_mesh, training, parallel_args)
            module.train()
    return modules, pp_schedule
