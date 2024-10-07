from dataclasses import dataclass


@dataclass  
class ParallelArgs:
    # dimension
    dp: int = 1
    tp: int = 1
    pp: int = 1
    
    # data parallel config
    dp_shard: bool = False
    compiled_autograd: bool = False
    mixed_precision_reduce: str = 'float32' 
    mixed_precision_param: str = 'float32'

    # tensor parallel config
    parallel_loss: bool = False
    async_tp: bool = False
    float8: bool = False

    # pipeline parallel config
    pipeline_parallel_schedule: str = 'gpipe'
    pipeline_parallel_microbatches: int = 2

    activation_checkpoint_mode: str = None
    selective_ac_option: str = 'op'
    compile: bool = False
    
    # rank
    dp_local_rank: int = 0
    dp_global_rank: int = 0
    tp_local_rank: int = 0
    tp_global_rank: int = 0
    pp_local_rank: int = 0
    pp_global_rank: int = 0

    # device
    device: str = "cuda:0"
