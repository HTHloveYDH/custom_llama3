from typing import Optional, List

import torch
from vllm.model_executor.models import Module
from vllm.model_executor.layers import Embedding, Linear, LayerNorm, ModuleList

from models.ModelArgs import ModelArgs
from models.modules import TransformerBlock, RMSNorm


class CustomModel(Module):
    def __init__(self, params:ModelArgs):
        super().__init__()
        self.params = params
        # wording embedding
        self.word_embeddings = Embedding(params.vocab_size, params.dim)
        # position embedding
        self.position_embeddings = Embedding(params.max_seq_len, params.dim)
        # Transformer Block layers
        self.layers = ModuleList([
            TransformerBlock(params) for _ in range(params.n_layers)
        ])
        # output normalization layer
        self.norm = RMSNorm(params.dim, params.eps) if params.norm_type == 'rmsnorm' else LayerNorm(params.dim)
        # output linear layer
        self.lm_head = Linear(params.dim, params.vocab_size, bias=False)
    
    def forward(self, x:torch.Tensor,
                position_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None):
        h = self.word_embeddings(x)
        if position_ids is not None:
            position_embeds = self.position_embeddings(position_ids)
            h = h + position_embeds
        for layer in self.layers:
            h = layer(h, attention_mask=attention_mask)
        h = self.norm(h)
        logits = self.lm_head(h)
        return logits
