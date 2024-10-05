import os
from collections import defaultdict

import torch

def convert(ckpt_path:str, format:str, save_dir:str, splits=3):
    import json

    from models.Transformer import Transformer
    
    def _convert_to_onnx(model:torch.nn.Module, save_dir:str):
        model.train()
        dummy_input = torch.randn(1, model.params.max_seq_len, dtype=torch.long)
        torch.onnx._export(
            model,
            dummy_input,
            os.path.join(save_dir, 'llama.onnx'),
            input_names=['input'],
            output_names=['output'],
            opset_version=13
        )

    def _convert_to_safetensors(model:torch.nn.Module, save_dir:str, splits:int):
        from safetensors.torch import save_file
        state_dict = model.state_dict()
        keys = list(state_dict.keys())
        T = len(keys) // splits
        for i in range(splits - 1):
            state_dict_split = OrderedDict()
            for j in range(i * T, (i + 1) * T):
                state_dict_split[keys[j]] = state_dict[keys[j]]
            save_file(state_dict_split, os.path.join(save_dir, f'./model_split_{i}.safetensors'))

    with open('./config/llama_config.json', 'r') as f:
        llama_config = json.load(f)
    ckpt = torch.load(ckpt_path)
    if hasattr(ckpt, 'model'):
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    llama = Transformer.from_local_pretrained(llama_config)
    llama.load_state_dict(state_dict)
    if format == 'onnx':
        _convert_to_onnx(llama)
    elif format == 'safetensors':
        _convert_to_safetensors(llama, splits)
