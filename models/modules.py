import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from config.ModelArgs import ModelArgs


class RoPE:
    @staticmethod
    def precompute_freqs_cis(dim:int, seq_len: int, theta: float=10000.0, device='cpu'):
        freqs = 1.0 / (theta** (torch.arange(0, dim, 2, device=device)[:(dim//2)].float()/dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, freqs).to(device)
        # The rotation matrix needs to be converted to polar form in order to
        # perform a rotation on the embedding
        freqs_cis = torch.polar(torch.ones_like(freqs).to(device), freqs).to(device)
        return freqs_cis

    @staticmethod
    def reshape_for_broadcast(freqs_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1],x.shape[-1]), "freqs_cis的最后两个维度必须与x匹配"
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    @staticmethod
    def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))    #xq_:[bsz, seq_len, n_heads, head_dim / 2]
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))    #xk_:[bsz, seq_len, n_heads, head_dim / 2]

        freqs_cis = RoPE.reshape_for_broadcast(freqs_cis, xq_)

        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)  # xq_out: [bsz, seq_len, n_heads, head_dim]
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)  # xk_out: [bsz, seq_len, n_heads, head_dim]
        return xq_out.type_as(xq), xk_out.type_as(xk)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float=1e-6):
        super().__init__()
        self.eps = eps
        # scale factor (namely, gamma)，defaults to torch.ones(dim)
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        # shape: x[bs,seq,dim]
        output = self._norm(x.float()).type_as(x)
        # shape: x[bs,seq,dim] -> x_norm[bs,seq,dim]
        return output * self.weight

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.n_rep = args.n_heads // args.n_kv_heads
        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

    def forward(self, x: torch.Tensor, start_pos):
        # x shape: [bsz, seq_len, dim]
        bsz, seq_len, _ = x.shape
        # causal mask is necessary in training mode
        # causal mask is necessary in inference mode due to kv cache.
        mask = None
        xq = self.wq(x)  # x [bsz, seq_len, dim] * wq [dim, n_heads * head_dim] -> q [bsz, seq_len, n_heads * head_dim]
        xk = self.wk(x)  # x [bsz, seq_len, dim] * wq [dim, n_kv_heads * head_dim] -> k [bsz, seq_len, n_kv_heads * head_dim]
        xv = self.wv(x)  # x [bsz, seq_len, dim] * wq [dim, n_kv_heads * head_dim] -> v [bsz, seq_len, n_kv_heads * head_dim]
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)  # xq [bsz, seq_len, n_heads, head_dim]
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)  # xk [bsz, seq_len, n_kv_heads, head_dim]
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)  # xv [bsz, seq_len,n_kv_heads, head_dim]
        # training mode: without kv-cache
        if self.training:
            # compute rotation matrix
            freqs_cis = RoPE.precompute_freqs_cis(
                self.head_dim, self.args.max_seq_len, self.args.rope_theta, x.device
            )
            # and apply RoPE to xq and xk
            # xq shape: [bsz,seq_len,n_heads, head_dim], xk shape: [bsz,seq_len,n_heads, head_dim]
            xq, xk = RoPE.apply_rotary_emb(xq, xk, freqs_cis)
            # shape: [bsz, seq_len, n_heads, head_dim]
            keys = Attention.repeat_kv(xk, self.n_rep)
            # shape: [bsz, seq_len, n_heads, head_dim]
            values = Attention.repeat_kv(xv, self.n_rep)
            # compute causal mask in training mode
            mask = torch.full((seq_len, seq_len), float("-inf"), device=x.device)
            mask = torch.triu(mask, diagonal=1).to(x)
        # inference mode: with kv-cache
        else:
            freqs_cis = RoPE.precompute_freqs_cis(
                self.head_dim, self.args.max_seq_len * 2, self.args.rope_theta, x.device
            )
            freqs_cis = freqs_cis[start_pos : start_pos + seq_len]
            # apply RoPE to query (xq) and value (xv)
            xq, xk = RoPE.apply_rotary_emb(xq, xk, freqs_cis)
            # shift cache_k, cache_v if necessary
            if start_pos % self.args.max_seq_len == 0:
                self.shift_cache()
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)
            # save key and value to cache (kv_cache)
            self.cache_k[:bsz, start_pos:start_pos + seq_len] = xk
            self.cache_v[:bsz, start_pos:start_pos + seq_len] = xv
            keys = self.cache_k[:bsz, :start_pos + seq_len]
            values = self.cache_v[:bsz, :start_pos + seq_len]
            keys = Attention.repeat_kv(keys, self.n_rep)  # keys shape: [bsz, seq_len, n_heads, head_dim]
            values = Attention.repeat_kv(values, self.n_rep)  # values shape: [bsz, seq_len, n_heads, head_dim]
        xq = xq.transpose(1, 2)  # xq shape: [bsz, n_heads, seq_len, head_dim]
        keys = keys.transpose(1, 2)  # keys shape: [bsz, n_heads,seq_len, head_dim]
        values = values.transpose(1, 2)  # values shape: [bsz, n_heads, seq_len, head_dim]
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        # shape: [bsz, n_heads, seq_len, head_dim] -> [bsz, seq_len, n_heads,head_dim] -> [bsz, seq_len, n_heads * head_dim]
        output = output.transpose(1,2).contiguous().view(bsz, seq_len, -1)
        return self.wo(output)  # shape: [bsz, seq_len, dim]

    def disable_kv_cache(self):
        # initialize cache to None (kv cache only for training mode)
        self.cache_k = None
        self.cache_v = None

    def enable_kv_cache(self):
        # initialize cache to store keys, values (kv cache only for inference mode)
        self.cache_k = torch.zeros(
            (self.args.max_batch_size, self.args.max_seq_len, self.n_kv_heads, self.head_dim),
            device=self.wq.weight.device
        )
        self.cache_v = torch.zeros(
            (self.args.max_batch_size, self.args.max_seq_len, self.n_kv_heads, self.head_dim),
            device=self.wq.weight.device
        )

    @staticmethod
    def repeat_kv(x:torch.Tensor, n_rep:int) -> torch.Tensor:
        bsz, seq_len, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        return (
            x[:, :, :, None, :]
            .expand(bsz, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(bsz, seq_len, n_kv_heads * n_rep, head_dim)
        )

    def shift_cache(self):
        self.cache_k[:, :-1, :, :] = self.cache_k[:, 1:, :, :]
        self.cache_v[:, :-1, :, :] = self.cache_v[:, 1:, :, :]

class FeedForward(nn.Module):
    def __init__(self, dim:int, hidden_dim:int, multiple_of:int, ffn_dim_multiplier: Optional[float]):
        super().__init__()
        self.dim = dim
        # hidden_dim should be divisable by 256
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(self.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, self.dim, bias=False)
        self.w3 = nn.Linear(self.dim, hidden_dim, bias=False)

    def forward(self, x):
        # shape: [bsz,seq_len,dim]
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.attention_norm = RMSNorm(dim=args.dim, eps=args.norm_eps)
        self.attention = Attention(args)
        self.ff_norm = RMSNorm(dim=args.dim, eps=args.norm_eps)
        self.feedforward=FeedForward(args.dim, 4*args.dim, args.multiple_of, args.ffn_dim_multiplier)

    def forward(self, x, start_pos):
        # start_pos: start posotion in inference mode
        # 1) input embedding to attention_norm,
        #    and attention_norm output is transfered to attention
        # 2) add attention output to embedding (before attention_norm)
        h = x + self.attention(self.attention_norm(x), start_pos)
        # 1) attention output is transfered to ff_norm
        #    then ff_norm output is transfered to feedforward
        # 2) add feedforward output to attention (before ff_norm)
        out = h + self.feedforward(self.ff_norm(h))
        # shape: [bsz, seq_len, dim]
        return out