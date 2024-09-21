import os

import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.device_mesh import init_device_mesh

from dist.device import get_devices


def init_dist(dist_type:str, dp_size:int, tp_size:int, *args):
    visible_devices = get_devices('cuda')
    print(f'{len(visible_devices)} visible devices: ', visible_devices, ' detected.')
    if dist_type in ['ddp', 'fsdp']:
        # use of FSDP or DDP demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), 'for now i think we need CUDA for DDP or FSDP'
        # launch by torch.multiprocessing
        torch_mp_launch = args[0]
        if torch_mp_launch:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            world_size = dp_size * tp_size
            local_rank = args[1]
            init_process_group(backend='nccl', rank=local_rank, world_size=world_size)
            dp_global_rank = dp_local_rank = global_rank = local_rank
        # lanuch by torchrun
        else:
            init_process_group(backend='nccl')
            global_rank = int(os.environ['RANK'])
            local_rank = int(os.environ['LOCAL_RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            assert world_size == dp_size * tp_size
            dp_global_rank = global_rank
            dp_local_rank = local_rank
        master_process = global_rank == 0 # this process will do logging, checkpointing etc.
        # added after video, pytorch can be serious about it's device vs. device_type distinction
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(device)
        print(f'using device: {device}')
        device_type = 'cuda' if device.startswith('cuda') else 'cpu'
        device_mesh = None
    elif dist_type in ['fsdp+tp', 'tp']:
        global_rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        assert world_size == dp_size * tp_size
        master_process = True
        # attempt to autodetect device
        device = 'cpu'  # defaults to 'cuda:0'
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        device_type = 'cuda' if device.startswith('cuda') else 'cpu'
        # Create a device mesh with 2 dimensions.
        # First dim is the data parallel dimension
        # Second dim is the tensor parallel dimension.
        device_mesh = init_device_mesh(device_type, (dp_size, tp_size), mesh_dim_names=('dp', 'tp'))
        dp_global_rank = device_mesh['dp'].get_rank()
        dp_local_rank = device_mesh['dp'].get_local_rank()
    elif dist_type == 'default':
        # vanilla, non-DDP run
        dp_global_rank = global_rank = 0
        dp_local_rank = local_rank = 0
        world_size = 1
        master_process = True
        # attempt to autodetect device
        device = 'cpu'  # defaults to 'cuda:0'
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        print(f'using device: {device}')
        device_type = 'cuda' if device.startswith('cuda') else 'cpu'
        device_mesh = None
    return dp_global_rank, dp_local_rank, device_mesh, master_process, device, device_type

def ternimate_dist(dist_type:str):
    if dist_type in ['ddp', 'fsdp']:
        destroy_process_group()
    else:
        pass