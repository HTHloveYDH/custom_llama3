'''reference url: 
1. https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/pipelining_utils.py
2. https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/pipeline_llama.py
'''
from typing import Tuple

from torch.distributed.pipelining import (
    Schedule1F1B,
    ScheduleFlexibleInterleaved1F1B,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
)

from models.Transformer import Transformer as Llama
from models.DPOLlama import DPOLlama


def build_pipeline_schedule(job_config, stages, loss_fn):
    looped_schedule = False

    if job_config.experimental.pipeline_parallel_schedule == "1f1b":
        schedule_class = Schedule1F1B
    elif job_config.experimental.pipeline_parallel_schedule == "gpipe":
        schedule_class = ScheduleGPipe
    elif job_config.experimental.pipeline_parallel_schedule == "interleaved_1f1b":
        schedule_class = ScheduleInterleaved1F1B
        looped_schedule = True
    # elif (
    #     job_config.experimental.pipeline_parallel_schedule
    #     == "flexible_interleaved_1f1b"
    # ):
    #     schedule_class = ScheduleFlexibleInterleaved1F1B
    #     looped_schedule = True
    else:
        raise NotImplementedError(
            f"{job_config.experimental.pipeline_parallel_schedule} is not implemented"
        )
    n_microbatches = job_config.experimental.pipeline_parallel_microbatches
    if n_microbatches is None:
        n_microbatches = job_config.experimental.pipeline_parallel_degree

    return schedule_class(
        stages if looped_schedule else stages[0],
        n_microbatches=n_microbatches,
        loss_fn=loss_fn,
    )


# TODO(whc) should this be a utility inside torch.pipelining?
def stage_ids_this_rank(
    pp_rank: int, pp_size: int, num_stages: int, style: str = "loop"
) -> Tuple[int]:
    """Compute the stage ids for the stages that will run on this pp rank for either a looped or V style schedule"""
    assert (
        num_stages % pp_size == 0
    ), f"num_stages {num_stages} must be evenly divisible by pp_size {pp_size}"
    stages_per_rank = num_stages // pp_size
    if style == "loop":
        return tuple(pp_rank + s * pp_size for s in range(stages_per_rank))
    elif style == "v":
        assert (
            stages_per_rank == 2
        ), f"v schedules assume 2 stages per rank, got {stages_per_rank}"
        stage_v_pairs = list(
            zip(range(pp_size), range(num_stages - 1, pp_size - 1, -1))
        )
        return stage_v_pairs[pp_rank]

def pipeline_parallelize_llama(model, pp_mesh):
    return model

def pipeline_parallelize(model, tp_mesh, training:bool):
    if isinstance(model, Llama):
        model = pipeline_parallelize_llama(model, tp_mesh, training)
    elif isinstance(model, DPOLlama):
        # TODO:
        model.llm = pipeline_parallelize_llama(model.llm, tp_mesh, training)
    return model
