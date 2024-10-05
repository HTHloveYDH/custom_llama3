'''reference url: 
1. https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/pipelining_utils.py
2. https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/pipeline_llama.py
'''
from typing import Callable, Union, Tuple
import copy

import torch
from torch.distributed import DeviceMesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining import (
    Schedule1F1B,
    # ScheduleFlexibleInterleaved1F1B,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
)

from config.torch_config import TORCH_DTYPE_MAP
from dist.ParallelArgs import ParallelArgs
from models.ModelArgs import ModelArgs
from models.Transformer import Transformer as Llama
from models.DPOLlama import DPOLlama
from utils.logging import logger


DeviceType = Union[int, str, torch.device]

def build_pipeline_schedule(parallel_args:ParallelArgs, stages, loss_fn):
    looped_schedule = False
    if parallel_args.pipeline_parallel_schedule == "1f1b":
        schedule_class = Schedule1F1B
    elif parallel_args.pipeline_parallel_schedule == "gpipe":
        schedule_class = ScheduleGPipe
    elif parallel_args.pipeline_parallel_schedule == "interleaved_1f1b":
        schedule_class = ScheduleInterleaved1F1B
        looped_schedule = True
    # elif (
    #     parallel_args.pipeline_parallel_schedule
    #     == "flexible_interleaved_1f1b"
    # ):
    #     schedule_class = ScheduleFlexibleInterleaved1F1B
    #     looped_schedule = True
    else:
        raise NotImplementedError(
            f"{parallel_args.pipeline_parallel_schedule} is not implemented"
        )
    n_microbatches = parallel_args.pipeline_parallel_microbatches
    if n_microbatches is None:
        n_microbatches = parallel_args.pipeline_parallel_degree

    return schedule_class(
        stages if looped_schedule else stages[0],
        n_microbatches=n_microbatches,
        loss_fn=loss_fn,
    )


# TODO(whc) should this be a utility inside torch.pipelining?
def stage_ids_this_rank(
    pp_local_rank:int, pp:int, num_stages:int, style:str = "loop"
) -> Tuple[int]:
    """Compute the stage ids for the stages that will run on this pp rank for either a looped or V style schedule"""
    assert (
        num_stages % pp == 0
    ), f"num_stages {num_stages} must be evenly divisible by pp_size {pp}"
    stages_per_rank = num_stages // pp
    if style == "loop":
        return tuple(pp_local_rank + s * pp for s in range(stages_per_rank))
    elif style == "v":
        assert (
            stages_per_rank == 2
        ), f"v schedules assume 2 stages per rank, got {stages_per_rank}"
        stage_v_pairs = list(
            zip(range(pp), range(num_stages - 1, pp - 1, -1))
        )
        return stage_v_pairs[pp_local_rank]

def _llama_trace_input(parallel_args:ParallelArgs, model_args:ModelArgs, device="meta"):
    """Get meta tensors with the right input shapes used for tracing"""
    tokens_shape = (model_args.max_batch_size, model_args.max_seq_len)
    tokens = torch.randint(
        model_args.vocab_size, tokens_shape, dtype=torch.int64, device=device
    )
    return (tokens,)


def _mixed_precision_dtype(
        parallel_args:ParallelArgs, 
        default:torch.dtype = torch.float32
    ) -> torch.dtype:
    """Get the mixed precision dtype if FSDP is enabled, otherwise return the default"""
    mppt = parallel_args.mixed_precision_param_type
    return TORCH_DTYPE_MAP[mppt] if parallel_args.dp > 1 else default


def pipeline_llama_manual_split(
        whole_model:nn.Module,
        pp_mesh:DeviceMesh,
        parallel_args:ParallelArgs,
        device:DeviceType,
        model_args:ModelArgs,
    ):
    """
    This API extracts one torch.nn.Module objects for the part of the model configured to run inside this stage.

    It wraps the model chunk in a ManualPipelineStage object and returns both the stage and model objects.

    The stage object is used to create a pipeline schedule, and the model object can be used for applying SPMD
    parallelism.
    """
    # pp_local_rank = pp_mesh.get_local_rank()
    pp_local_rank = parallel_args.pp_local_rank
    # pp = pp_mesh.size()
    pp = parallel_args.pp
    microbatches = (
        parallel_args.pipeline_parallel_microbatches or parallel_args.pp > 1
    )
    splits = parallel_args.pipeline_parallel_split_points

    def _build_stage(stage_idx, start_layer, stop_layer, is_first=False, is_last=False):
        model = copy.deepcopy(whole_model)
        if not is_first:
            model.tok_embeddings = None
        drop_layers = start_layer is not None
        for name in list(model.layers.keys()):
            # we keep layers in a contiguous region between start (inclusive) and stop (exclusive)
            if f"layers.{name}" == start_layer:
                drop_layers = False
            if f"layers.{name}" == stop_layer:
                drop_layers = True
            if drop_layers:
                del model.layers[name]

        if not is_last:
            model.norm = None
            model.output = None

        # Note: these tensors are only here as metadata hints, so pipelining runtime knows what size buffer to allocate.
        # these tensors should be on meta device, adn the model should also.  It will be allocated on device after
        # applying all other parallelisms.

        # TODO(whc) once ManualPipelineStage supports lazy shape inference, we can avoid specifying input/output shapes
        mp_dtype = _mixed_precision_dtype(parallel_args, parallel_args)
        max_batch_size = model_args.max_batch_size
        local_seq_len = int(model_args.max_seq_len // parallel_args.tp)
        layers_io_shape = (max_batch_size, local_seq_len, model_args.dim)
        output_layer_shape = (
            max_batch_size,
            model_args.max_seq_len,
            model_args.vocab_size,
        )
        if is_first:
            (input,) = _llama_trace_input(parallel_args, model_args, device="meta")
        else:
            # later layers (assume all start w/ a transformer layer)
            input = torch.rand(layers_io_shape, dtype=mp_dtype, device="meta")
        if is_last:
            output = torch.rand(output_layer_shape, dtype=torch.float32, device="meta")
        else:
            # earlier layers (assume all end in a transformer layer)
            output = torch.rand(layers_io_shape, dtype=mp_dtype, device="meta")
        stage = PipelineStage(
            model,
            stage_idx,
            num_stages,
            device,
            input_args=input.chunk(microbatches)[0],
            output_args=output.chunk(microbatches)[0],
            group=pp_mesh.get_group("pp"),
        )
        return stage, model

    num_stages = len(splits) + 1
    stage_idx = pp_local_rank
    stages = []
    models = []
    for stage_idx in stage_ids_this_rank(pp_local_rank, pp, num_stages, style="loop"):
        start_layer = splits[stage_idx - 1] if stage_idx > 0 else None
        stop_layer = splits[stage_idx] if stage_idx < num_stages - 1 else None
        stage, model_chunk = _build_stage(
            stage_idx,
            start_layer,
            stop_layer,
            is_first=stage_idx == 0,
            is_last=stage_idx == num_stages - 1,
        )
        logger.info(
            f"PP rank {pp_local_rank} is building stage_idx {stage_idx}"
            f" with start_layer {start_layer}, stop_layer {stop_layer}: model chunk \n{model_chunk}"
        )
        stages.append(stage)
        models.append(model_chunk)
    return stages, models

def pipeline_parallelize_llama(
        model:nn.Module,
        pp_mesh:DeviceMesh,
        parallel_args:ParallelArgs,
        device:DeviceType,
        model_args:ModelArgs,
        loss_fn:Callable[..., torch.Tensor],):
    stages, models = pipeline_llama_manual_split(
        model, pp_mesh, parallel_args, parallel_args, device, model_args
    )
    pp_schedule = build_pipeline_schedule(parallel_args, stages, loss_fn)
    return pp_schedule, models

def pipeline_parallelize(model:nn.Module, pp_mesh:DeviceMesh, training:bool):
    if isinstance(model, Llama):
        model = pipeline_parallelize_llama(model, pp_mesh, training)
    elif isinstance(model, DPOLlama):
        # TODO:
        model.llm = pipeline_parallelize_llama(model.llm, pp_mesh, training)
    return model
