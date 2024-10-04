'''reference url: 
1. https://github.com/pytorch/torchtitan/blob/main/torchtitan/utils.py
2. https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py#L274
'''
import os
from collections import defaultdict, OrderedDict

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

from dist.logging import logger


def get_num_params(model: torch.nn.Module, exclude_embedding: bool = False) -> int:
    num_params = sum(p.numel() for p in model.parameters())
    if exclude_embedding:
        num_params -= model.tok_embeddings.weight.numel()
    return num_params

def get_num_flop_per_token(num_params: int, model_config, seq_len) -> int:
    l, h, q, t = (
        model_config.n_layers,
        model_config.n_heads,
        model_config.dim // model_config.n_heads,
        seq_len,
    )
    # Reasoning behind the factor of 12 for the self-attention part of the formula:
    # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
    # 2. the flash attention does 1 more matmul recomputation in the backward
    #    but recomputation should not be counted in calculating MFU           (+0)
    # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
    # 4. we follow the convention and do not account for sparsity in causal attention
    flop_per_token = 6 * num_params + 12 * l * h * q * t

    return flop_per_token

def apply_compile(model: torch.nn.Module):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = torch.compile(transformer_block, fullgraph=True)
        model.layers.register_module(layer_id, transformer_block)

    logger.info("Compiling each TransformerBlock with torch.compile")

# for selective op activation checkpointing
_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
}

def _apply_ac_to_transformer_block(module: nn.Module, ac_config):
    valid_ac_modes = ("full", "selective")
    if ac_config.mode not in valid_ac_modes:
        raise ValueError(
            f"Invalid AC mode: {ac_config.mode}. Valid modes: {valid_ac_modes}"
        )

    if ac_config.mode == "full":
        return ptd_checkpoint_wrapper(module, preserve_rng_state=False)

    assert ac_config.mode == "selective", f"{ac_config.mode}"
    use_op_sac = ac_config.selective_ac_option == "op"
    use_layer_sac = ac_config.selective_ac_option.isdigit()
    if not use_op_sac and not use_layer_sac:
        raise ValueError(
            f"Invalid selective AC option: {ac_config.selective_ac_option}. "
            f"Valid options: 'op' or a positive int representing layer frequency"
        )
    if use_op_sac:
        from torch.utils.checkpoint import (
            CheckpointPolicy,
            create_selective_checkpoint_contexts,
        )

        def _get_custom_policy(meta):
            def _custom_policy(ctx, func, *args, **kwargs):
                mode = "recompute" if ctx.is_recompute else "forward"
                mm_count_key = f"{mode}_mm_count"
                if func == torch.ops.aten.mm.default:
                    meta[mm_count_key] += 1
                # Saves output of all compute ops, except every second mm
                to_save = func in _save_list and not (
                    func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0
                )
                return (
                    CheckpointPolicy.MUST_SAVE
                    if to_save
                    else CheckpointPolicy.PREFER_RECOMPUTE
                )

            return _custom_policy

        def selective_checkpointing_context_fn():
            meta = defaultdict(int)
            return create_selective_checkpoint_contexts(_get_custom_policy(meta))

        return ptd_checkpoint_wrapper(
            module,
            context_fn=selective_checkpointing_context_fn,
            preserve_rng_state=False,
        )
    elif use_layer_sac:
        # Checkpoint every `ac_freq` of the modules passed to this function
        ac_freq = int(ac_config.selective_ac_option)
        ptd_checkpoint_wrapper.__dict__.setdefault("_count", 0)
        ptd_checkpoint_wrapper._count += 1
        if not ac_freq or ptd_checkpoint_wrapper._count % ac_freq == 0:
            return ptd_checkpoint_wrapper(module, preserve_rng_state=False)
        else:
            return module

def apply_activation_checkpoint(model: torch.nn.Module, ac_config):
    """Apply activation checkpointing to the model."""
    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = _apply_ac_to_transformer_block(transformer_block, ac_config)
        model.layers.register_module(layer_id, transformer_block)

    logger.info(f"Applied {ac_config.mode} activation checkpointing to the model")

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
