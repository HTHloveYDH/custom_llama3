import torch
import torch.nn as nn
from torch.nn import functional as F

from config.ModelArgs import ModelArgs
from models.modules import RMSNorm, Attention, TransformerBlock


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()    
        self.params = params   
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)  
        self.layers = nn.ModuleList()  
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(args=params))    
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)    
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)  
 
    def forward(self, x, targets=None, start_pos=0):
        # start_pos: start posotion in inference mode
        # x shape: [bsz, seq_len] -> h shape: [bsz, seq_len, dim]  
        h = self.tok_embeddings(x)     
        for layer in self.layers:  
            h = layer(h, start_pos)   
        h = self.norm(h)
        # self.output maps embedding to logits with length of vocabulary  
        # h shape: [bsz, seq_len, dim] -> logits shape: [bsz, seq_len, vocab_size]  
        logits = self.output(h).float()  
        loss = None   
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.params.vocab_size), targets.view(-1))  
        return logits, loss

    @staticmethod
    def create_llama_model(llama_config:dict):
        args_map = {
            'llama3_8B': dict(
                dim=4096, n_layers=32, n_heads=32, n_kv_heads=8, vocab_size=128256, 
                multiple_of=1024, ffn_dim_multiplier=1.3, norm_eps=1e-5, rope_theta=500000.0, 
                max_batch_size=32, max_seq_len=2048
            ),  # 8B parameters
            'llama3_70B': dict(
                dim=8192, n_layers=80, n_heads=64, n_kv_heads=8, vocab_size=128256, 
                multiple_of=4096, ffn_dim_multiplier=1.3, norm_eps=1e-5, rope_theta=500000.0, 
                max_batch_size=32, max_seq_len=2048
            ),  # 70B parameters
            'llama3_405B': dict(
                dim=16384, n_layers=126, n_heads=128, n_kv_heads=8, vocab_size=128256, 
                multiple_of=None, ffn_dim_multiplier=1.3, norm_eps=1e-5, rope_theta=500000.0, 
                max_batch_size=32, max_seq_len=2048
            ),  # 405B parameters
        }.get(llama_config['model_type'], {})
        # create llama3 by custom configuration
        if not args_map:
            args_map.update(llama_config['params'])
            assert args_map['n_heads'] % args_map['n_kv_heads'] == 0, f'make sure n_heads is divisible by n_kv_heads'
        model = Transformer(ModelArgs(**args_map))
        return model

    @classmethod
    def from_official_pretrained(cls, llama_config:dict):
        from safetensors.torch import load_file  # pip install safetensors
        assert llama_config['model_type'] in ['llama3_8B', 'llama3_70B', 'llama3_405B'], f"{llama_config['model_type']} is invalid"
        model = Transformer.create_llama_model(llama_config)
        loaded_tensors = load_file(llama_config['ckpt_path'], device='cpu')
        model.load_state_dict(loaded_tensors)
        return model

    @classmethod
    def from_local_pretrained(cls, llama_config:dict):
        model = Transformer.create_llama_model(llama_config)
        ckpt = torch.load(llama_config['ckpt_path'])
        model.load_state_dict(ckpt)
        return model
    
    @classmethod
    def from_scratch(cls, llama_config:dict):
        model = Transformer.create_llama_model(llama_config)
        return model

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'IS_INIT_SCALE'):
                std *= (2 * self.params.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)  # 1 / sqrt(768) = 0.036, sqrt(1600) = 0.025
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    @staticmethod
    def init_lora_weights(module, rank=1, alpha=1):
        '''
        usage: 
            import functools
            init_lora_weights = functools.partial(Transformer.init_lora_weights, rank=rank, alpha=alpha)
            llama3 = Transformer(ModelArgs(**args_map))
            llama3.apply(init_lora_weights)
        '''
        if isinstance(module, nn.Linear):
            LoRAParametrization.inject_lora_weights(module, rank=1, alpha=1)
    
    def init_lora(self, rank=1, alpha=1):
        '''
        usage: 
            llama3 = Transformer(ModelArgs(**args_map))
            llama3.init_lora(rank=rank, alpha=alpha)
        '''
        # original_weights = LoRAParametrization.get_original_weights(self)
        original_non_lora_weights = LoRAParametrization.count_original_non_lora_weights(self)
        for module in self.modules():
            self.init_lora_weights(module, rank, alpha)
        _, _ = LoRAParametrization.count_lora_weights(self, original_non_lora_weights)
        # LoRAParametrization.confirm_original_weights(self, original_weights)
        # LoRAParametrization.enable_disable_lora(self, True)
        LoRAParametrization.freeze_non_lora_weights(self)

    def train(self, mode: bool = True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, Attention):
                module.disable_kv_cache()


    def eval(self):
        super().eval()
        for module in self.modules():
            if isinstance(module, Attention):
                module.enable_kv_cache()