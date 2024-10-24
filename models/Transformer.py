import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributed.tensor.parallel import loss_parallel

from models.ModelArgs import ModelArgs
from models.modules import RoPE, RMSNorm, Attention, InfiniteAttention, TransformerBlock, \
    IdentityAttention, IdentityMLP
from models.lora import LoRAParametrization


class Transformer(nn.Module):
    def __init__(self, params:ModelArgs):
        super().__init__()
        self.params = params
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.layers = nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(args=params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

    def forward(self, x, start_pos=0):
        # start_pos: start posotion in inference mode
        # x shape: [bsz, seq_len] -> h shape: [bsz, seq_len, dim]
        h = self.tok_embeddings(x)
        self.freqs_cis = self.freqs_cis.to(h.device)
        for layer in self.layers:
            h = layer(h, start_pos, self.freqs_cis)
        h = self.norm(h)
        # self.output maps embedding to logits with length of vocabulary
        # h shape: [bsz, seq_len, dim] -> logits shape: [bsz, seq_len, vocab_size]
        logits = self.output(h).float()
        return logits

    def compute_loss(self, pred, target, tp:bool, parallel_loss:bool):
        if target is not None:
            if tp:
                if parallel_loss:
                    with loss_parallel():
                        loss = F.cross_entropy(pred.view(-1, self.params.vocab_size), target.view(-1))
                else:
                    loss = F.cross_entropy(pred.view(-1, pred.size(-1)), target.view(-1))
            else:
                loss = F.cross_entropy(pred.view(-1, self.params.vocab_size), target.view(-1))
        return loss

    def precompute_freqs_cis(self, mode:bool):
        # compute rotation matrix
        if mode:
            self.freqs_cis = RoPE.precompute_freqs_cis(
                self.params.dim // self.params.n_heads, self.params.max_seq_len,
                self.params.rope_theta
            )
        else:
            # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096.
            # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
            self.freqs_cis = RoPE.precompute_freqs_cis(
                self.params.dim // self.params.n_heads, self.params.max_seq_len * 2,
                self.params.rope_theta
            )  # (use 2x max sequence length to be safe)

    @classmethod
    def create_llama_model(cls, llama_config:dict):
        args_map = {
            'llama2_7B': dict(
                dim=4096, n_layers=32, n_heads=32, n_kv_heads=32, vocab_size=32000,
                multiple_of=256, ffn_dim_multiplier=1.3, norm_eps=1e-5, rope_theta=10000.0,
                max_batch_size=32, max_seq_len=2048, long_term_memory=False, norm_type='rmsnorm'
            ),  # 7B parameters
            'llama2_13B': dict(
                dim=5120, n_layers=40, n_heads=40, n_kv_heads=40, vocab_size=32000,
                multiple_of=256, ffn_dim_multiplier=1.3, norm_eps=1e-5, rope_theta=10000.0,
                max_batch_size=32, max_seq_len=2048, long_term_memory=False, norm_type='rmsnorm'
            ),  # 13B parameters
            'llama2_70B': dict(
                dim=8192, n_layers=80, n_heads=64, n_kv_heads=8, vocab_size=32000,
                multiple_of=4096, ffn_dim_multiplier=1.3, norm_eps=1e-5, rope_theta=10000.0,
                max_batch_size=32, max_seq_len=2048, long_term_memory=False, norm_type='rmsnorm'
            ),  # 70B parameters
            'llama3_8B': dict(
                dim=4096, n_layers=32, n_heads=32, n_kv_heads=8, vocab_size=128256,
                multiple_of=1024, ffn_dim_multiplier=1.3, norm_eps=1e-5, rope_theta=500000.0,
                max_batch_size=32, max_seq_len=2048, long_term_memory=False, norm_type='rmsnorm'
            ),  # 8B parameters
            'llama3_70B': dict(
                dim=8192, n_layers=80, n_heads=64, n_kv_heads=8, vocab_size=128256,
                multiple_of=4096, ffn_dim_multiplier=1.3, norm_eps=1e-5, rope_theta=500000.0,
                max_batch_size=32, max_seq_len=2048, long_term_memory=False, norm_type='rmsnorm'
            ),  # 70B parameters
            'llama3_405B': dict(
                dim=16384, n_layers=126, n_heads=128, n_kv_heads=8, vocab_size=128256,
                multiple_of=None, ffn_dim_multiplier=1.3, norm_eps=1e-5, rope_theta=500000.0,
                max_batch_size=32, max_seq_len=2048, long_term_memory=False, norm_type='rmsnorm'
            ),  # 405B parameters
        }.get(llama_config['model_type'], {})
        # create llama3 by custom configuration
        if not args_map:
            args_map.update(llama_config['params'])
            assert args_map['n_heads'] % args_map['n_kv_heads'] == 0, f'make sure n_heads is divisible by n_kv_heads'
        return cls(ModelArgs(**args_map))

    @staticmethod
    def from_official_pretrained(llama_config:dict):
        from safetensors import safe_open  # pip install safetensors
        assert llama_config['model_type'] in [
            'llama2_7B', 'llama2_13B', 'llama2_70B', 
            'llama3_8B', 'llama3_70B', 'llama3_405B'
        ]
        model = Transformer.create_llama_model(llama_config)
        state_dict = {}
        weight_files = [
            os.path.join(llama_config['ckpt_path'], file)
            for file in os.listdir(llama_config['ckpt_path'])
            if 'safetensors' in file
        ]
        for weight_file in weight_files:
            # supported values for framework:'pt', 'tf', 'flax', 'numpy'
            with safe_open(weight_file, framework='pt', device='cpu') as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        model.load_state_dict(state_dict)
        return model

    @staticmethod
    def from_local_pretrained(llama_config:dict):
        from collections import OrderedDict
        model = Transformer.create_llama_model(llama_config)
        if llama_config['lora']:
            assert llama_config['lora_ckpt_path'] is not None
            model.init_lora(llama_config['lora_rank'], llama_config['lora_alpha'])
            ckpt = torch.load(llama_config['ckpt_path'])
            lora_ckpt = torch.load(llama_config['lora_ckpt_path'])
            full_state_dict = OrderedDict()
            full_state_dict.update(ckpt['model'])
            full_state_dict.update(lora_ckpt['model'])
            model.load_state_dict(full_state_dict)
            return model
        ckpt = torch.load(llama_config['ckpt_path'])
        model.load_state_dict(ckpt)
        return model

    @staticmethod
    def from_scratch(llama_config:dict):
        model = Transformer.create_llama_model(llama_config)
        if llama_config['lora']:
            model.init_lora(rank=llama_config['lora_rank'], alpha=llama_config['lora_alpha'])
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

    def _drop_layers(self, attn_list:list, mlp_list:list):
        attn_list.sort()
        for i, idx in enumerate(attn_list):
            self.layers[idx - i].attention = IdentityAttention()
        mlp_list.sort()
        for i, idx in enumerate(mlp_list):
            self.layers[idx - i].feedforward = IdentityMLP()

    def _drop_blocks(self, blocks:list):
        for i in blocks:
            self.layers.pop(i)

    def drop_modules(self, attn_list:list, mlp_list:list):
        block_list = list(set(mlp_list) & set(attn_list))  # common_indices
        mlp_list = list(set(mlp_list) - set(block_list))
        attn_list = list(set(attn_list) - set(block_list))
        self._drop_blocks(block_list)
        self._drop_layers(attn_list, mlp_list)

    def train(self, mode:bool=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, (Attention, InfiniteAttention)):
                if mode:
                    module.disable_kv_cache()
                else:
                    module.enable_kv_cache()
        self.precompute_freqs_cis(mode)

    def eval(self):
        super().eval()
        for module in self.modules():
            if isinstance(module, (Attention, InfiniteAttention)):
                module.enable_kv_cache()
        self.precompute_freqs_cis(False)
