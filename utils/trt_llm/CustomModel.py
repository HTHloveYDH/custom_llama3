import tensorrt_llm
from tensorrt_llm import Module, Tensor
from tensorrt_llm import functional as F
from tensorrt_llm.layers import Embedding, Linear, LayerNorm, ModuleList

from utils.trt_llm.custom_modules import RMSNorm, TransformerBlock
from models.ModelArgs import ModelArgs


class CustomModel(Module):
    def __init__(self, params:ModelArgs):
        super().__init__()
        self.embed_dim = params.dim
        self.num_layers = params.n_layers
        # embedding layer
        self.embeddings = Embedding(
            num_embeddings=params.vocab_size,
            embedding_dim=params.dim
        )
        # Transformer Block layers
        self.layers = ModuleList([
            TransformerBlock(params) for _ in range(params.n_layers)
        ])
        # output normalization layer
        self.norm = RMSNorm(params.dim, params.eps) if params.norm_type == 'rmsnorm' else LayerNorm(params.dim)
        # output linear layer
        self.lm_head = Linear(params.dim, params.vocab_size)
    
    def forward(self, x:Tensor, position_ids=None):
        h = self.embeddings(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        logits = self.lm_head(h)
        return logits
