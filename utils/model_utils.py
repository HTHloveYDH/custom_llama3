import os
import json
import re
from collections import OrderedDict

import torch
import torch.nn as nn


# rules
replacement_rules = {
    r'^tok_embeddings\.weight$': 'model.embed_tokens.weight',
    r'^layers\.(\d+)\.attention_norm\.weight$': r'model.layers.\1.input_layernorm.weight',
    r'^layers\.(\d+)\.attention\.wq\.weight$': r'model.layers.\1.self_attn.q_proj.weight',
    r'^layers\.(\d+)\.attention\.wk\.weight$': r'model.layers.\1.self_attn.k_proj.weight',
    r'^layers\.(\d+)\.attention\.wv\.weight$': r'model.layers.\1.self_attn.v_proj.weight',
    r'^layers\.(\d+)\.attention\.wo\.weight$': r'model.layers.\1.self_attn.o_proj.weight',
    r'^layers\.(\d+)\.ff_norm\.weight$': r'model.layers.\1.post_attention_layernorm.weight',
    r'^layers\.(\d+)\.feedforward\.w1\.weight$': r'model.layers.\1.mlp.gate_proj.weight',
    r'^layers\.(\d+)\.feedforward\.w2\.weight$': r'model.layers.\1.mlp.down_proj.weight',
    r'^layers\.(\d+)\.feedforward\.w3\.weight$': r'model.layers.\1.mlp.up_proj.weight',
    r'^norm\.weight$': 'model.norm.weight',
    r'^output\.weight$': 'lm_head.weight'
}

def replace_key(key):
    for pattern, replacement in replacement_rules.items():
        match = re.match(pattern, key)
        if match:
            if len(match.groups()) > 0:
                return re.sub(pattern, replacement, key)
            else:
                return replacement
    return key

def convert(ckpt_path:str, format:str, save_dir:str, splits=4):
    from models.Transformer import Transformer
    
    def _convert_to_onnx(model:nn.Module, save_dir:str):
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

    def _convert_to_safetensors(state_dict:OrderedDict, save_dir:str, splits:int):
        from safetensors.torch import save_file
        json_content = {'metadata': {'total_size': None}, 'weight_map': {}}
        keys = list(state_dict.keys())
        T = len(keys) // splits
        # for first N - 1 splits
        for i in range(splits - 1):
            state_dict_split = OrderedDict()
            for j in range(i * T, (i + 1) * T):
                json_content['weight_map'][replace_key(keys[j])] = f'model-0000{i + 1}-of-0000{splits}.safetensors'
                state_dict_split[replace_key(keys[j])] = state_dict[keys[j]]
            metadata = {'format': 'pt'}
            save_file(
                state_dict_split, os.path.join(save_dir, f'model-0000{i + 1}-of-0000{splits}.safetensors'), 
                metadata=metadata
            )
            torch.save(
                state_dict_split, os.path.join(save_dir, f'pytorch_model-0000{i + 1}-of-0000{splits}.bin')
            )
        # for last split
        state_dict_split = OrderedDict()
        for j in range((splits - 1) * T, len(keys)):
            json_content['weight_map'][replace_key(keys[j])] = f'model-0000{splits}-of-0000{splits}.safetensors'
            state_dict_split[replace_key(keys[j])] = state_dict[keys[j]]
        metadata = {'format': 'pt'}
        save_file(
            state_dict_split, os.path.join(save_dir, f'model-0000{splits}-of-0000{splits}.safetensors'), 
            metadata=metadata
        )
        torch.save(
            state_dict_split, os.path.join(save_dir, f'pytorch_model-0000{splits}-of-0000{splits}.bin')
        )
        # create model.safetensors.index.json
        with open(os.path.join(save_dir, 'model.safetensors.index.json'), 'w') as f:
            json.dump(json_content, f)

    with open('./config/llama_config.json', 'r') as f:
        llama_config = json.load(f)
    ckpt = torch.load(ckpt_path)
    if hasattr(ckpt, 'model'):
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    if format == 'onnx':
        llama = Transformer.from_local_pretrained(llama_config)
        llama.load_state_dict(state_dict)
        _convert_to_onnx(llama, save_dir)
    elif format == 'safetensors':
        _convert_to_safetensors(state_dict, save_dir, splits)
