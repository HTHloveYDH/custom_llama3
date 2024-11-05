from tensorrt_llm import Module

from models.ModelArgs import ModelArgs


class RMSNorm(Module):
    def __init__(self, dim:int, eps:float=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
    
    def forward(self, x):
        return x

class TransformerBlock(Module):
    def __init__(self, params:ModelArgs):
        super().__init__()
        self.params = params
    
    def forward(self, x, start_pos:int, freqs_cis):
        return x