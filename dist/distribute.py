import os

import torch
from torch.distributed import init_process_group, destroy_process_group

from dist.device import get_devices


def init_dist(dist_strategy:str, torch_mp_launch:bool, dp_local_rank:int, dp_world_size:int):
    visible_devices = get_devices('cuda')
    print(f'{len(visible_devices)} visible devices: ', visible_devices, ' detected.')
    if dist_strategy in ['ddp', 'fsdp']:
        # use of FSDP or DDP demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), 'for now i think we need CUDA for DDP or FSDP'
        # launch by torch.multiprocessing
        if torch_mp_launch:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            init_process_group(backend='nccl', rank=dp_local_rank, world_size=dp_world_size)
            dp_global_rank = dp_local_rank
        # lanuch by torchrun
        else:
            init_process_group(backend='nccl')
            dp_global_rank = int(os.environ['RANK'])
            dp_local_rank = int(os.environ['LOCAL_RANK'])
            dp_world_size = int(os.environ['WORLD_SIZE'])
        master_process = dp_global_rank == 0 # this process will do logging, checkpointing etc.
        # added after video, pytorch can be serious about it's device vs. device_type distinction
        device = f'cuda:{dp_local_rank}'
        torch.cuda.set_device(device)
        print(f"using device: {device}")
    elif dist_strategy in ['default']:
        # vanilla, non-DDP run
        dp_global_rank = 0
        dp_local_rank = 0
        dp_world_size = 1
        master_process = True
        # attempt to autodetect device
        device = "cpu"  # defaults to 'cuda:0'
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")
    device_type = 'cuda' if device.startswith('cuda') else 'cpu'
    return dp_global_rank, dp_local_rank, dp_world_size, master_process, device, device_type

def ternimate_dist(dist_strategy:str):
    if dist_strategy in ['ddp', 'fsdp']:
        destroy_process_group()
    else:
        pass