import torch


def get_devices(device_type='cuda'):
    devices = []
    if device_type == 'cuda':
        device_num = torch.cuda.device_count()
    elif device_type == 'cpu':
        device_num = torch.cpu.device_count()
    else:
        raise ValueError(f" {device_type} is not supported ")
    for i in range(0, device_num):
        devices.append(torch.cuda.get_device_name(i))
    return devices

def get_visible_devices(visible_device_indices:list, device_type='cuda'):
    visible_devices = []
    devices = get_devices(device_type=device_type)
    if len(visible_device_indices) > len(devices):
        raise RuntimeError(f" assigned devices exceed existing devices ")
    if device_type == 'cuda':
        for i in visible_device_indices:
            visible_devices.append(devices[i])
        # print(': ', torch.cuda.current_device())  # torch.cuda.current_device() returns '0'
    elif device_type == 'cpu':
        visible_devices.append('cpu')
    print('current devices: ', visible_devices, f'   ({device_type})')
    return visible_devices