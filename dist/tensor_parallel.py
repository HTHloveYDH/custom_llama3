import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed._tensor import Shard, Replicate
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    SequenceParallel
)

from dist.ParallelArgs import ParallelArgs
from models.Transformer import Transformer as Llama
from models.DPOLlama import DPOLlama
from utils.logging import logger


def tensor_parallelize_llama(model:nn.Module, tp_mesh:DeviceMesh, training:bool, parallel_args:ParallelArgs):
    # parallelize the first embedding and the last linear out projection
    layer_tp_plan = {
        'tok_embeddings': RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
        ),
        'norm': SequenceParallel(),
        'output': ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Shard(-1) if parallel_args.parallel_loss else Replicate(), 
            use_local_output=not parallel_args.parallel_loss
        ),
    }
    parallelize_module(model, tp_mesh, layer_tp_plan)
    for block_id, transformer_block in enumerate(model.layers):
        layer_tp_plan = {
            'attention_norm': SequenceParallel(),
            'attention': PrepareModuleInput(
                input_layouts=(Shard(1), None, None),
                desired_input_layouts=(Replicate(), None, None),
            ),
            'attention.wq': ColwiseParallel(),
            'attention.wk': ColwiseParallel(),
            'attention.wv': ColwiseParallel(),
            'attention.wo': RowwiseParallel(output_layouts=Shard(1)),
            'ffn_norm': SequenceParallel(),
            'feed_forward': PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            'feed_forward.w1': ColwiseParallel(),
            'feed_forward.w2': RowwiseParallel(output_layouts=Shard(1)),
            'feed_forward.w3': ColwiseParallel(),
        }
        # Adjust attention module to use the local number of heads
        attn_layer = transformer_block.attention
        attn_layer.n_heads = attn_layer.n_heads // tp_mesh.size()
        attn_layer.n_kv_heads = attn_layer.n_kv_heads // tp_mesh.size()
        # Custom parallelization plan for the model
        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_tp_plan
        )
    if parallel_args.async_tp:
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group
        torch._inductor.config._micro_pipeline_tp = True
        enable_symm_mem_for_group(tp_mesh.get_group().group_name)
    logger.info(
        f"Applied {'Float8 ' if parallel_args.float8 else ''}{'Async ' if parallel_args.async_tp else ''}"
        "Tensor Parallelism to the model"
    )
    return model

def tensor_parallelize(model:nn.Module, tp_mesh:DeviceMesh, training:bool, parallel_args:ParallelArgs):
    assert not (parallel_args.parallel_loss and not training)
    if isinstance(model, Llama):
        model = tensor_parallelize_llama(model, tp_mesh, training, parallel_args)
    elif isinstance(model, DPOLlama):
        # TODO:
        assert training
        layer_tp_plan = {'value_head': ColwiseParallel()}
        model = parallelize_module(model, tp_mesh, layer_tp_plan)
        model.llm = tensor_parallelize_llama(model.llm, tp_mesh, training, parallel_args)
    return model
