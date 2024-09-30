from torch.distributed._tensor import Shard, Replicate
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    SequenceParallel
)

from models.Transformer import Transformer as Llama
from models.DPOLlama import DPOLlama


def llama_TP(model, tp_mesh, training:bool, parallel_loss:bool):
    assert not (parallel_loss and not training)
    # parallelize the first embedding and the last linear out projection
    layer_tp_plan = {
        'tok_embeddings': RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
        ),
        'norm': SequenceParallel(),
        'output': ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Replicate()
        ),
    }
    model = parallelize_module(model, tp_mesh, layer_tp_plan)
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
    return model

def TP(model, tp_mesh, training:bool):
    if isinstance(model, Llama):
        model = llama_TP(model, tp_mesh, training)
    elif isinstance(model, DPOLlama):
        # TODO:
        assert training
        layer_tp_plan = {'value_head': ColwiseParallel()}
        model = parallelize_module(model, tp_mesh, layer_tp_plan)
        model.llm = llama_TP(model.llm, tp_mesh, training)
    return model
