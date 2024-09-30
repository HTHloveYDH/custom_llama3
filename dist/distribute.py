import os

import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.device_mesh import init_device_mesh

from dist.device import get_devices

def create_device_mesh(parallel_dims:dict, device_type:str):
    device_mesh = None
    dim_names = []
    dims = []
    for key in parallel_dims.keys():
        if parallel_dims[key] > 1:
            dim_names.append(key)
            dims.append(parallel_dims[key])
    device_mesh = init_device_mesh(device_type, dims, mesh_dim_names=dim_names)
    return device_mesh

def init_dist(parallel_dims:dict):
    visible_devices = get_devices('cuda')
    print(f'{len(visible_devices)} visible devices: ', visible_devices, ' detected.')
    if parallel_dims['dp'] == 1 and parallel_dims['tp'] == 1 and parallel_dims['pp'] == 1:
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
        return master_process, device, device_type, device_mesh
    init_process_group(backend='nccl')
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE']) 
    assert world_size == parallel_dims['dp'] * parallel_dims['tp'] * parallel_dims['pp']
    master_process = global_rank == 0 # this process will do logging, checkpointing etc.
    # added after video, pytorch can be serious about it's device vs. device_type distinction
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)
    print(f'using device: {device}')
    device_type = 'cuda' if device.startswith('cuda') else 'cpu'
    device_mesh = create_device_mesh(parallel_dims, device_type)
    return master_process, device, device_type, device_mesh

def ternimate_dist(parallel_dims:dict):
    if parallel_dims['dp'] > 1 or parallel_dims['tp'] > 1 or parallel_dims['pp'] > 1:
        destroy_process_group()