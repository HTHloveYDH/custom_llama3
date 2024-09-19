from dataclasses import dataclass


@dataclass  
class ModelArgs:  
    dim: int = 512                    # embedding dimensions
    n_layers: int = 8                 # number of transformer block
    n_heads: int = 8                  # number of query heads  
    n_kv_heads: int = 4               # number of key / value heads  
    vocab_size: int = 128256          # vocabulary size  
    multiple_of: int = 256            # for feedforward network  
    ffn_dim_multiplier: float = 1.3   # for feedforward network  
    norm_eps: float = 1e-5            # default Epsilon of RMSNorm
    rope_theta: float = 10000.0       # default theta of RoPE  

    max_batch_size: int = 10          # max batch size  
    max_seq_len: int = 256            # max seqence length  

    long_term_memory: bool = False