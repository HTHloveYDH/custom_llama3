from dataclasses import dataclass


@dataclass  
class ParallelArgs:
    # dimension
    dp: int = 1
    tp: int = 1
    pp: int = 1
    
    # data parallel config
    dp_shard: bool = False

    # tensor parallel config
    parallel_loss: bool = False
    async_tp: bool = False
    float8: bool = False

    # pipeline parallel config
    pipeline_parallel_schedule: str = "gpipe"

    activation_checkpoint_mode: str = None
    compile: bool = False
    
    # rank
    dp_local_rank: int = 0
    dp_global_rank: int = 0
    tp_local_rank: int = 0
    tp_global_rank: int = 0
    pp_local_rank: int = 0
    pp_global_rank: int = 0
