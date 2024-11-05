import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.ModelArgs import ModelArgs
from models.TritonFusedRMSNorm import TritonFusedRMSNorm


class RoPE:
    @staticmethod
    def precompute_freqs_cis(dim:int, seq_len:int, theta:float=10000.0, device='cpu'):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[:(dim // 2)].float() / dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, freqs).to(device)
        # The rotation matrix needs to be converted to polar form in order to
        # perform a rotation on the embedding
        freqs_cis = torch.polar(torch.ones_like(freqs).to(device), freqs).to(device)
        return freqs_cis

    @staticmethod
    def reshape_for_broadcast(freqs_cis:torch.Tensor, x:torch.Tensor):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1]), 'Last two dimensions of freqs_cis need to be compatible with x.'
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    @staticmethod
    def apply_rotary_emb(xq:torch.Tensor, xk:torch.Tensor, freqs_cis:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # xq_ shape: [bsz, seq_len, n_heads, head_dim / 2]
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))  # xk_ shape: [bsz, seq_len, n_heads, head_dim / 2]

        freqs_cis = RoPE.reshape_for_broadcast(freqs_cis, xq_)

        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)  # xq_out shape: [bsz, seq_len, n_heads, head_dim]
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)  # xk_out shape: [bsz, seq_len, n_heads, head_dim]
        return xq_out.type_as(xq), xk_out.type_as(xk)

class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float=1e-6):
        super().__init__()
        self.eps = eps
        # scale factor (namely, gamma)，defaults to torch.ones(dim)
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x:torch.Tensor):
        # shape: x[bs,seq,dim]
        output = self._norm(x.float()).type_as(x)
        # shape: x[bs,seq,dim] -> x_norm[bs,seq,dim]
        return output * self.weight
    
    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore

class FusedRMSNorm(nn.Module):
    """Fused RMS Norm, wraps a fused Triton Kernel"""

    def __init__(self, dim:int, eps:float=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return TritonFusedRMSNorm.apply(x, self.weight, self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """leverages Triton Fused RMS Norm kernel"""
        output = self._norm(x.float()).type_as(x)
        return output

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore

class Attention(nn.Module):
    def __init__(self, args:ModelArgs):
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

    def forward(self, x:torch.Tensor, start_pos:int, freqs_cis:torch.Tensor):
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
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)  # xv [bsz, seq_len, n_kv_heads, head_dim]
        # training mode: without kv-cache
        if self.training:
            # and apply RoPE to xq and xk
            # xq shape: [bsz,seq_len, n_heads, head_dim], xk shape: [bsz,seq_len, n_heads, head_dim]
            xq, xk = RoPE.apply_rotary_emb(xq, xk, freqs_cis)
            # shape: [bsz, seq_len, n_heads, head_dim]
            keys = Attention.repeat_kv(xk, self.n_rep)
            # shape: [bsz, seq_len, n_heads, head_dim]
            values = Attention.repeat_kv(xv, self.n_rep)
            # compute causal mask in training mode
            mask = torch.full((seq_len, seq_len), float('-inf'), device=x.device)
            mask = torch.triu(mask, diagonal=1).to(x)
        # inference mode: with kv-cache
        else:
            # freqs_cis_ = freqs_cis[start_pos:start_pos + seq_len]
            start_pos_ = start_pos % (self.args.max_seq_len * 2)
            freqs_cis_ = freqs_cis[start_pos_:start_pos_ + seq_len]
            # apply RoPE to query (xq) and value (xv)
            xq, xk = RoPE.apply_rotary_emb(xq, xk, freqs_cis_)
            # shift cache_k, cache_v if necessary
            if start_pos % self.args.max_seq_len == 0:
                self._shift_cache()
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)
            # save key and value to cache (kv_cache)
            self.cache_k[:bsz, start_pos:start_pos + seq_len] = xk
            self.cache_v[:bsz, start_pos:start_pos + seq_len] = xv
            keys = self.cache_k[:bsz, :start_pos + seq_len]
            values = self.cache_v[:bsz, :start_pos + seq_len]
            keys = Attention.repeat_kv(keys, self.n_rep)  # keys shape: [bsz, seq_len, n_heads, head_dim]
            values = Attention.repeat_kv(values, self.n_rep)  # values shape: [bsz, seq_len, n_heads, head_dim]
        # compute attention score matrix
        xq = xq.transpose(1, 2)  # xq shape: [bsz, n_heads, seq_len, head_dim]
        keys = keys.transpose(1, 2)  # keys shape: [bsz, n_heads,seq_len, head_dim]
        values = values.transpose(1, 2)  # values shape: [bsz, n_heads, seq_len, head_dim]
        # normal attention
        output = self._normal_scaled_dot_product_attention(xq, keys, values, mask) # output shape: [bsz, n_heads, seq_len, head_dim]
        # # flash attention (more efficient)
        # output = F.scaled_dot_product_attention(xq, keys, values, attn_mask=mask)  # output shape: [bsz, n_heads, seq_len, head_dim]
        # shape: [bsz, n_heads, seq_len, head_dim] -> [bsz, seq_len, n_heads,head_dim] -> [bsz, seq_len, n_heads * head_dim]
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.wo(output)  # shape: [bsz, seq_len, dim]

    def _normal_scaled_dot_product_attention(self, q, k, v, mask):
        attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn.float(), dim=-1).type_as(q)
        output = torch.matmul(attn, v)
        return output

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

    def _shift_cache(self):
        # self.cache_k[:, :-1, :, :] = self.cache_k[:, 1:, :, :]  # O(seq_len), not efficient, sometimes error happens
        # self.cache_v[:, :-1, :, :] = self.cache_v[:, 1:, :, :]  # O(seq_len), not efficient, sometimes error happens
        self.cache_k = torch.cat(
            [
                self.cache_k[:, 1:, :, :],
                torch.zeros(
                    (self.args.max_batch_size, 1, self.n_kv_heads, self.head_dim),
                    device=self.cache_k.device
                )
            ],
            dim=1
        )  # O(1), efficient
        self.cache_v = torch.cat(
            [
                self.cache_v[:, 1:, :, :],
                torch.zeros(
                    (self.args.max_batch_size, 1, self.n_kv_heads, self.head_dim),
                    device=self.cache_v.device
                )
            ],
            dim=1
        )  # O(1), efficient

class InfiniteAttention(Attention):
    def __init__(self, args:ModelArgs):
        super().__init__(args)
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

        self.beta = nn.Parameter(torch.randn(1, self.n_heads, 1, 1))
        self.register_buffer(
            'M', torch.zeros(self.args.max_batch_size, self.n_heads, self.head_dim, self.head_dim)
        )
        self.register_buffer(
            'z', torch.zeros(self.args.max_batch_size, self.n_heads, self.head_dim, 1)
        )

    def forward(self, x:torch.Tensor, start_pos:int, freqs_cis:torch.Tensor):
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
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)  # xv [bsz, seq_len, n_kv_heads, head_dim]
        # training mode: without kv-cache
        if self.training:
            # and apply RoPE to xq and xk
            # xq shape: [bsz,seq_len, n_heads, head_dim], xk shape: [bsz,seq_len, n_heads, head_dim]
            xq, xk = RoPE.apply_rotary_emb(xq, xk, freqs_cis)
            # shape: [bsz, seq_len, n_heads, head_dim]
            keys = Attention.repeat_kv(xk, self.n_rep)
            # shape: [bsz, seq_len, n_heads, head_dim]
            values = Attention.repeat_kv(xv, self.n_rep)
            # compute causal mask in training mode
            mask = torch.full((seq_len, seq_len), float('-inf'), device=x.device)
            mask = torch.triu(mask, diagonal=1).to(x)
        # inference mode: with kv-cache
        else:
            # freqs_cis_ = freqs_cis[start_pos:start_pos + seq_len]
            start_pos_ = start_pos % (self.args.max_seq_len * 2)
            freqs_cis_ = freqs_cis[start_pos_:start_pos_ + seq_len]
            # apply RoPE to query (xq) and value (xv)
            xq, xk = RoPE.apply_rotary_emb(xq, xk, freqs_cis_)
            # shift cache_k, cache_v if necessary
            if start_pos % self.args.max_seq_len == 0:
                self._shift_cache()
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)
            # save key and value to cache (kv_cache)
            self.cache_k[:bsz, start_pos:start_pos + seq_len] = xk
            self.cache_v[:bsz, start_pos:start_pos + seq_len] = xv
            keys = self.cache_k[:bsz, :start_pos + seq_len]
            values = self.cache_v[:bsz, :start_pos + seq_len]
            keys = InfiniteAttention.repeat_kv(keys, self.n_rep)  # keys shape: [bsz, seq_len, n_heads, head_dim]
            values = InfiniteAttention.repeat_kv(values, self.n_rep)  # values shape: [bsz, seq_len, n_heads, head_dim]
        # compute attention score matrix
        xq = xq.transpose(1, 2)  # xq shape: [bsz, n_heads, seq_len, head_dim]
        keys = keys.transpose(1, 2)  # keys shape: [bsz, n_heads, seq_len, head_dim]
        values = values.transpose(1, 2)  # values shape: [bsz, n_heads, seq_len, head_dim]
        # normal attention
        output = self._normal_scaled_dot_product_attention(xq, keys, values, mask)  # output shape: [bsz, n_heads, seq_len, head_dim]
        # # flash attention (more efficient)
        # output = F.scaled_dot_product_attention(xq, keys, values, attn_mask=mask)  # output shape: [bsz, n_heads, seq_len, head_dim]
        # Memory retrieval and attention calculation per segment
        # retrieve memory
        retrieved_memory = self._retrieve_from_memory(xq, self.M, self.z)
        # Update memory with current segment's key and value states
        self.M, self.z  = self._update_memory(keys, values, self.M, self.z)
        output = self._long_term_mem_injection(output, retrieved_memory)
        # shape: [bsz, n_heads, seq_len, head_dim] -> [bsz, seq_len, n_heads, head_dim] -> [bsz, seq_len, n_heads * head_dim]
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.wo(output)  # shape: [bsz, seq_len, dim]

    def _retrieve_from_memory(self, q, M, z):
        # Retrieve context from compressive memory using linear attention (Eq. 3)
        M_s_1 = torch.matmul(F.elu(q) + 1, M)  # shape: [bsz, n_heads, seq_len, head_dim]
        Z_s_1 = torch.matmul(F.elu(q) + 1, z) + 1e-8  # shape: [bsz, n_heads, seq_len, 1]
        A_mem = M_s_1 / Z_s_1  # shape: [bsz, n_heads, seq_len, head_dim]
        return A_mem

    def _update_memory(self, k, v, M, z, use_delta=False):
        if use_delta:
            retrieved_v = torch.matmul(F.elu(k) + 1, M) / \
                (torch.matmul(F.elu(k) + 1, z.unsqueeze(-1)) + 1e-8)
            M = M + torch.matmul(F.elu(k).transpose(-2, -1) + 1, v - retrieved_v)
        else:
            M = M + torch.matmul(F.elu(k).transpose(-2, -1) + 1, v)
        z = z + (F.elu(k) + 1).sum(dim=-2, keepdim=True).transpose(-2, -1)
        return M, z

    def _long_term_mem_injection(self, output, retrieved_memory):
        beta = torch.sigmoid(self.beta)
        return beta * retrieved_memory + (1 - beta) * output

    def reset_memory(self):
        self.M.zero_()
        self.z.zero_()

class FeedForward(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()
        self.args = args
        # hidden_dim should be divisable by 256
        hidden_dim = int(2 * 4 * args.dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x:torch.Tensor):
        # shape: [bsz,seq_len,dim]
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

'''
reference url: https://github.com/cooper12121/llama3-8x8b-MoE/blob/main/modeling_file/modeling_llama_moe.py#L750
'''
class MoEFeedForward(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, args:ModelArgs):
        super().__init__()
        assert args.moe_top_k <= args.n_experts
        self.args = args
        # gating
        self.gate = nn.Linear(args.dim, args.n_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(args) for _ in range(args.n_experts)])

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        B, T, dim = x.shape
        x = x.view(-1, dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(x)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.args.moe_top_k, dim=-1)  # [B * T, moe_top_k], expert index
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)  # 这是相当于百分比？归一化？
        # we cast back to the input dtype
        routing_weights = routing_weights.to(x.dtype)
        x = torch.zeros((B * T, dim), dtype=x.dtype, device=x.device)
        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.args.n_experts).permute(2, 1, 0)
        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.args.n_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.shape[0] == 0:
                continue
            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = x[None, top_x_list].reshape(-1, dim)
            current_expert_x = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]
            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            x.index_add_(0, top_x, current_expert_x.to(x.dtype))
        x = x.reshape(B, T, dim)
        # return x, router_logits
        return x

class TransformerBlock(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()
        self.args = args
        self.attention_norm = RMSNorm(dim=args.dim, eps=args.norm_eps)
        if args.long_term_memory:
            self.attention = InfiniteAttention(args)
        else:
            self.attention = Attention(args)
        if args.norm_type == 'fused_rmsnorm':
            self.ff_norm = FusedRMSNorm(dim=args.dim, eps=args.norm_eps)
        else:
            self.ff_norm = RMSNorm(dim=args.dim, eps=args.norm_eps)
        if args.n_experts > 0:
            self.feedforward = MoEFeedForward(args)
        else:
            self.feedforward = FeedForward(args)

    def forward(self, x:torch.Tensor, start_pos:int, freqs_cis:torch.Tensor):
        # start_pos: start posotion in inference mode
        # 1) input embedding to attention_norm,
        #    and attention_norm output is transfered to attention
        # 2) add attention output to embedding (before attention_norm)
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis)
        # 1) attention output is transfered to ff_norm
        #    then ff_norm output is transfered to feedforward
        # 2) add feedforward output to attention (before ff_norm)
        out = h + self.feedforward(self.ff_norm(h))
        # shape: [bsz, seq_len, dim]
        return out

class IdentityAttention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x:torch.Tensor, start_pos:int, freqs_cis:torch.Tensor):
        return x

class IdentityMLP(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x:torch.Tensor):
        return x
