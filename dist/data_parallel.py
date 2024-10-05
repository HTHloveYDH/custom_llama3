'''reference url: https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py'''
from collections import defaultdict

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed._composable.replicate import replicate

from config.torch_config import TORCH_DTYPE_MAP
from dist.ParallelArgs import ParallelArgs
from models.Transformer import Transformer as Llama
from models.DPOLlama import DPOLlama
from utils.logging import logger


def _check_strided_sharding_enabled() -> None:
    # Correct 2D/3D DCP usage requires DTensor's strided sharding in PR
    # https://github.com/pytorch/pytorch/pull/130760. This function checks if users'
    # PyTorch nightly-build version is newer than 2024-08-09 to make sure this PR is
    # included when 2D/3D DCP is used.
    if "git" in torch.__version__:  # pytorch is built from source
        # notify users to check if the commit hash is newer than 2024-08-09
        logger.warning(
            "detected that the pytorch is built from source. Please make sure the PR "
            "(https://github.com/pytorch/pytorch/pull/130760) is included in pytorch "
            "for correct 2D/3D DCP usage."
        )
    elif torch.__version__ < "2.5.0.dev20240809":
        # the nightly build pytorch was built before 2024-08-09
        logger.warning(
            f"detected that the pytorch version {torch.__version__} is older than "
            "2.5.0.dev20240809. Please upgrade a newer version to include the change "
            "made in https://github.com/pytorch/pytorch/pull/130760 for correct 2D/3D "
            "DCP usage."
        )

def enable_fsdp(
        model:nn.Module,
        dp_mesh:DeviceMesh,
        param_dtype:torch.dtype,
        reduce_dtype:torch.dtype,
        tp_enabled:bool,
        pp_enabled:bool,
    ):
    """
    Apply data parallelism to the model. FSDP2 is used here.
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

    # TODO: remove this check once PyTorch 2.5 is released. We can safely assume
    # that users won't use a nightly build which is older than 20240809 by then.
    if tp_enabled:
        # check if strided sharding is enabled, which is necessary for 2D/3D DCP
        _check_strided_sharding_enabled()

    for layer_id, transformer_block in model.layers.items():
        if pp_enabled:
            # For PP, do not reshard after forward to avoid per-microbatch
            # all-gathers, which can be expensive and non-overlapped
            reshard_after_forward = False
        else:
            # As an optimization, do not reshard after forward for the last
            # transformer block since FSDP would prefetch it immediately
            reshard_after_forward = int(layer_id) < len(model.layers) - 1
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    fully_shard(model, **fsdp_config, reshard_after_forward=not pp_enabled)


def enable_ddp(
        model:nn.Module,
        dp_mesh:DeviceMesh,
        enable_compile:bool,
        enable_compiled_autograd:bool,
    ):
    if enable_compile:
        if enable_compiled_autograd:
            torch._dynamo.config.optimize_ddp = (
                "python_reducer_without_compiled_forward"
            )
        else:
            torch._dynamo.config.optimize_ddp = "ddp_optimizer"

    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)

    logger.info("Applied DDP to the model")

def data_parallelize_llama(model:nn.Module, dp_mesh:DeviceMesh, training:bool, parallel_args:ParallelArgs):
    if parallel_args.dp_shard:
        enable_fsdp(
            model, dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[parallel_args.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[parallel_args.mixed_precision_reduce],
            tp_enabled=parallel_args.tp > 1, pp_enabled=parallel_args.pp > 1
        )
    else:
        enable_ddp(
            model, dp_mesh,
            enable_compile=parallel_args.compile,
            enable_compiled_autograd=parallel_args.compiled_autograd,
        )

def data_parallelize(model:nn.Module, dp_mesh:DeviceMesh, training:bool, parallel_args:ParallelArgs):
    if isinstance(model, Llama):
        data_parallelize_llama(model, dp_mesh, training, parallel_args)
    elif isinstance(model, DPOLlama):
        # TODO:
        data_parallelize_llama(model.llm, dp_mesh, training, parallel_args)
