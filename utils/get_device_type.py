def get_device_type(device:str):
    if 'cpu' in device:
        return 'cpu'
    if 'cuda' in device:
        return 'cuda'