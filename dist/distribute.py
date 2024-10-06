import os

import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed import DeviceMesh

from dist.ParallelArgs import ParallelArgs
from dist.device import get_devices


def _create_device_mesh(dist:dict, device_type:str):
    device_mesh = None
    dim_names = []
    dims = []
    for key in ['dp', 'tp', 'pp']:
        if dist[key] > 1:
            dim_names.append(key)
            dims.append(dist[key])
    device_mesh = init_device_mesh(device_type, dims, mesh_dim_names=dim_names)
    return device_mesh

def _get_ranks(dist:dict, device_mesh:DeviceMesh):
    ranks = {}
    for key in ['dp', 'tp', 'pp']:
        if dist[key] > 1:
            ranks[key + 'local_rank'] = device_mesh[key].get_local_rank()
            ranks[key + 'global_rank'] = device_mesh[key].get_rank()
        else:
            ranks[key + 'local_rank'] = 0
            ranks[key + 'global_rank'] = 0
    return ranks

def init_dist(dist:dict):
    visible_devices = get_devices('cuda')
    print(f'{len(visible_devices)} visible devices: ', visible_devices, ' detected.')
    if dist['dp'] == 1 and dist['tp'] == 1 and dist['pp'] == 1:
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
        dist.update({'device': device})
        parallel_args = ParallelArgs(**dist)
        return master_process, device, device_mesh, parallel_args
    init_process_group(backend='nccl')
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE']) 
    assert world_size == dist['dp'] * dist['tp'] * dist['pp']
    master_process = global_rank == 0  # this process will do logging, checkpointing etc.
    # added after video, pytorch can be serious about it's device vs. device_type distinction
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)
    print(f'using device: {device}')
    device_type = 'cuda' if device.startswith('cuda') else 'cpu'
    device_mesh = _create_device_mesh(dist, device_type)
    dist.update(_get_ranks(dist, device_mesh))
    dist.update({'device': device})
    parallel_args = ParallelArgs(**dist)
    return master_process, device, device_mesh, parallel_args

def ternimate_dist(dist:dict):
    if dist['dp'] > 1 or dist['tp'] > 1 or dist['pp'] > 1:
        destroy_process_group()
