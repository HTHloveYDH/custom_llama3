import os
import sys
import json

import torch
import torch.nn as nn

sys.path.append(os.getcwd())
from data_pipeline.Tokenizer import Tokenizer, ChatFormat
from models.ModelArgs import ModelArgs
from models.modules import RoPE, RMSNorm, Attention, InfiniteAttention, FeedForward, MoEFeedForward, \
    TransformerBlock
from models.Transformer import Transformer
from utils.model_utils import replace_key
# from data_pipeline.demo import DemoDataLoader


def RoPETEST(model_args:ModelArgs):
    for device in ['cpu', 'cuda:0']:
        x_norm = torch.randn(
            (model_args.max_batch_size, model_args.max_seq_len, model_args.dim),
            device=device
        )
        head_dim = model_args.dim // model_args.n_heads
        wq = nn.Linear(model_args.dim, model_args.n_heads * head_dim, bias=False, device=device)
        wk = nn.Linear(model_args.dim, model_args.n_kv_heads * head_dim, bias=False, device=device)
        xq = wq(x_norm)
        xk = wk(x_norm)
        print(f'xq.shape: {xq.shape}')
        print(f'xk.shape: {xk.shape}')
        xq = xq.view(xq.shape[0], xq.shape[1],model_args.n_heads, head_dim)
        xk = xk.view(xk.shape[0], xk.shape[1],model_args.n_kv_heads, head_dim)
        print(f'xq.re-shape: {xq.shape}')
        print(f'xk.re-shape: {xk.shape}')
        freqs_cis = RoPE.precompute_freqs_cis(head_dim, model_args.max_seq_len, model_args.rope_theta, device)
        print(f'freqs_cis.shape: {freqs_cis.shape}')
        xq_rotate, xk_rotate = RoPE.apply_rotary_emb(xq, xk, freqs_cis)
        print(f'[RoPE] xq_rotate.shape: {xq_rotate.shape}')
        print(f'[RoPE] xk_rotate.shape: {xk_rotate.shape}')
        print(f'[RoPE] RoPETEST on device: {device} passed')

def RMSNormTEST(model_args:ModelArgs):
    rms_norm = RMSNorm(dim=model_args.dim)
    for device in ['cpu', 'cuda:0']:
        x = torch.randn(
            (model_args.max_batch_size, model_args.max_seq_len, model_args.dim),
            device=device
        )
        rms_norm.to(device)
        x_norm = rms_norm(x)
        print(f'[RMSNorm] x_norm.shape: {x_norm.shape}')
        print(f'[RMSNorm] RMSNormTEST on device: {device} passed')

def AttentionTEST(model_args:ModelArgs):
    for device in ['cpu', 'cuda:0']:
        head_dim = model_args.dim // model_args.n_heads
        x_norm = torch.randn(
            (model_args.max_batch_size, model_args.max_seq_len, model_args.dim),
            device=device
        )
        xk = torch.randn(
            (model_args.max_batch_size, model_args.max_seq_len, model_args.n_kv_heads, head_dim),
            device=device
        )
        n_rep = model_args.n_heads // model_args.n_kv_heads
        keys = Attention.repeat_kv(xk, n_rep)
        print(f'xk.shape: {xk.shape}')
        print(f'keys.shape: {keys.shape}')
        attention = Attention(model_args)
        attention.to(device)
        freqs_cis = RoPE.precompute_freqs_cis(
            attention.args.dim // attention.args.n_heads, 
            attention.args.max_seq_len, attention.args.rope_theta
        )  # (use 2x max sequence length to be safe)
        x_out = attention(x_norm, start_pos=0, freqs_cis=freqs_cis.to(device))
        print(f'[Attention] x_out.shape: {x_out.shape}')
        print(f'[Attention] AttentionTEST on device: {device} passed')

def InfiniteAttentionTEST(model_args:ModelArgs):
    for device in ['cpu', 'cuda:0']:
        head_dim = model_args.dim // model_args.n_heads
        x_norm = torch.randn(
            (model_args.max_batch_size, model_args.max_seq_len, model_args.dim),
            device=device
        )
        xk = torch.randn(
            (model_args.max_batch_size, model_args.max_seq_len, model_args.n_kv_heads, head_dim),
            device=device
        )
        n_rep = model_args.n_heads // model_args.n_kv_heads
        keys = InfiniteAttention.repeat_kv(xk, n_rep)
        print(f'xk.shape: {xk.shape}')
        print(f'keys.shape: {keys.shape}')
        attention = InfiniteAttention(model_args)
        attention.to(device)
        freqs_cis = RoPE.precompute_freqs_cis(
            attention.args.dim // attention.args.n_heads, 
            attention.args.max_seq_len, attention.args.rope_theta
        )  # (use 2x max sequence length to be safe)
        x_out = attention(x_norm, start_pos=0, freqs_cis=freqs_cis.to(device))
        attention.reset_memory()
        print(f'[InfiniteAttention] x_out.shape: {x_out.shape}')
        print(f'[InfiniteAttention] InfiniteAttentionTEST on device: {device} passed')

def FeedForwardTEST(model_args:ModelArgs):
    for device in ['cpu', 'cuda:0']:
        x_out = torch.randn(
            (model_args.max_batch_size, model_args.max_seq_len, model_args.dim),
            device=device
        )
        feed_forward = FeedForward(model_args)
        feed_forward.to(device)
        rms_norm = RMSNorm(dim=model_args.dim)
        rms_norm.to(device)
        x_out = rms_norm(x_out)
        x_out = feed_forward(x_out)
        print(f'[FeedForward] x_out.shape: {x_out.shape}')
        print(f'[FeedForward] FeedForwardTEST on device: {device} passed')

def MoEFeedForwardTEST(model_args:ModelArgs):
    model_args.n_experts = 4
    for device in ['cpu', 'cuda:0']:
        x_out = torch.randn(
            (model_args.max_batch_size, model_args.max_seq_len, model_args.dim),
            device=device
        )
        feed_forward = MoEFeedForward(model_args)
        feed_forward.to(device)
        rms_norm = RMSNorm(dim=model_args.dim)
        rms_norm.to(device)
        x_out = rms_norm(x_out)
        x_out = feed_forward(x_out)
        print(f'[MoEFeedForward] x_out.shape: {x_out.shape}')
        print(f'[MoEFeedForward] MoEFeedForwardTEST on device: {device} passed')

def TransformerBlockTEST(model_args:ModelArgs):
    for device in ['cpu', 'cuda:0']:
        for n_experts in [-1, 4]:
            model_args.n_experts = n_experts
            x = torch.randn((model_args.max_batch_size, model_args.max_seq_len, model_args.dim), device=device)
            transformer_block = TransformerBlock(model_args)
            transformer_block.to(device)
            freqs_cis = RoPE.precompute_freqs_cis(
                transformer_block.args.dim // transformer_block.args.n_heads, 
                transformer_block.args.max_seq_len, transformer_block.args.rope_theta
            )  # (use 2x max sequence length to be safe)
            transformer_block_out = transformer_block(x, start_pos=0, freqs_cis=freqs_cis.to(device))
            print(f'[TransformerBlock] transformer_block_out.shape: {transformer_block_out.shape}')
            print(f'[TransformerBlock] TransformerBlockTEST on device: {device} passed')

def TransformerTEST(model_args:ModelArgs):
    for device in ['cpu', 'cuda:0']:
        model = Transformer(model_args).to(device)
        print('[Transformer] ', model)
        print(f'[Transformer] TransformerTEST on device: {device} passed')

# def DemoDataLoaderTEST(model_args:ModelArgs):
#     data_loader = DemoDataLoader('./data/demo/txt', model_args.max_seq_len, model_args.max_batch_size, 'full')
#     prompts = "Hello World"
#     encoded_tokens, _ = data_loader.encode(prompts)
#     decoded_text = data_loader.decode(encoded_tokens)
#     print(f"Shakespeare text length: {data_loader.vocab_size}")
#     print(f"vocabulary content: {''.join(data_loader.vocab)}\n")
#     print(f"vocabulary size: {data_loader.vocab_size}")
#     print(f"encode result: {encoded_tokens}")
#     print(f"decode result: {decoded_text}")
#     print(f'[DemoDataLoader] DemoDataLoaderTEST passed')

def TokenizerTEST():
    tokenizer_dir = os.path.join('.', 'tokenizer', 'llama3')
    for tokenizer_file in os.listdir(tokenizer_dir):
        tokenizer = Tokenizer(os.path.join(tokenizer_dir, tokenizer_file))
        tokens, _ = tokenizer.encode("This is a test sentence.", bos=True, eos=True)
        print(f"encode result: {tokens}")
        text = tokenizer.decode([128000, 2028, 374, 264, 1296, 11914, 13, 128001])
        print(f"decode result: {text}")
    print(f'[Tokenizer] TokenizerTEST passed')

def ChatFormatTEST():
    message = {
        "role": "user",
        "content": "This is a test sentence.",
    }
    dialog = [
        {
            "role": "system",
            "content": "This is a test sentence.",
        },
        {
            "role": "user",
            "content": "This is a response.",
        }
    ]
    tokenizer_dir = os.path.join('.', 'tokenizer', 'llama3')
    for tokenizer_file in os.listdir(tokenizer_dir):
        tokenizer = Tokenizer(os.path.join(tokenizer_dir, tokenizer_file))
        chat_format = ChatFormat(tokenizer)
        tokens = chat_format.encode_message(message)
        print(f"encode message result: {tokens}")
        text = tokenizer.decode(tokens)
        print(f"decode result: {text}")
        tokens = chat_format.encode_dialog_prompt(dialog)
        print(f"encode dialog prompt: {tokens}")
        text = tokenizer.decode(tokens)
        print(f"decode result: {text}")
    print(f'[ChatFormat] ChatFormatTEST passed')

def RegularityTEST():
    input_file = './test/model.safetensors.index.json' 
    output_file = './test/converted_model.safetensors.index.json' 
    with open(input_file, 'r') as f:
        data = json.load(f)
    new_data = {
        'metadata': data['metadata'],
        'weight_map': {}
    }
    for key, value in data['weight_map'].items():
        new_key = replace_key(key)
        new_data['weight_map'][new_key] = value
    with open(output_file, 'w') as f:
        json.dump(new_data, f, indent=2)
    print(f'[RegularityTEST] RegularityTEST passed')

def DropModulesTEST():
    for device in ['cpu', 'cuda:0']:
        model = Transformer(model_args).to(device)
        print('[Drop Modules] before drop: ', model)
        model.drop_modules(attn_list=[0, 2, 3], mlp_list=[0, 3])
        print('[Drop Modules] after drop: ', model)
        print(f'[Drop Modules] DropModulesTEST on device: {device} passed')


if __name__ == '__main__':
    model_args = ModelArgs()
    RoPETEST(model_args)
    RMSNormTEST(model_args)
    AttentionTEST(model_args)
    InfiniteAttentionTEST(model_args)
    FeedForwardTEST(model_args)
    MoEFeedForwardTEST(model_args)
    TransformerBlockTEST(model_args)
    TransformerTEST(model_args)
    # DemoDataLoaderTEST(model_args)
    TokenizerTEST()
    ChatFormatTEST()
    RegularityTEST()
    DropModulesTEST()
    print('\n')
    print('==================== all tests passed ====================')
