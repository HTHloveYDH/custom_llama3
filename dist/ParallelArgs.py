from dataclasses import dataclass


@dataclass  
class ParallelArgs:
    # dimension
    dp: int = 1
    tp: int = 1
    pp: int = 1
    
    # other configs
    dp_shard: bool = False

    parallel_loss: bool = False
    async_tp: bool = False
    float8: bool = False

    pipeline_parallel_schedule: str = "gpipe"

    # rank
    dp_local_rank: int = 0
    dp_global_rank: int = 0
    tp_local_rank: int = 0
    tp_global_rank: int = 0
    pp_local_rank: int = 0
    pp_global_rank: int = 0