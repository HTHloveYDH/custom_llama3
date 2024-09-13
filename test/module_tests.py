import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.getcwd())
from data_pipeline.Tokenizer import Tokenizer, ChatFormat
from config.ModelArgs import ModelArgs
from models.modules import RoPE, RMSNorm, Attention, FeedForward, TransformerBlock
from models.Transformer import Transformer
# from data_pipeline.demo import DemoDataLoader


def RoPETEST(arg_map):
    for device in ['cpu', 'cuda:0']:
        x_norm = torch.randn(
            (arg_map.max_batch_size, arg_map.max_seq_len, arg_map.dim),
            device=device
        )
        head_dim = arg_map.dim // arg_map.n_heads
        wq = nn.Linear(arg_map.dim, arg_map.n_heads * head_dim, bias=False, device=device)
        wk = nn.Linear(arg_map.dim, arg_map.n_kv_heads * head_dim, bias=False, device=device)
        xq = wq(x_norm)
        xk = wk(x_norm)
        print(f'xq.shape: {xq.shape}')
        print(f'xk.shape: {xk.shape}')
        xq = xq.view(xq.shape[0], xq.shape[1],arg_map.n_heads, head_dim)
        xk = xk.view(xk.shape[0], xk.shape[1],arg_map.n_kv_heads, head_dim)
        print(f'xq.re-shape: {xq.shape}')
        print(f'xk.re-shape: {xk.shape}')
        freqs_cis = RoPE.precompute_freqs_cis(head_dim, arg_map.max_seq_len, arg_map.rope_theta, device)
        print(f'freqs_cis.shape: {freqs_cis.shape}')
        xq_rotate, xk_rotate = RoPE.apply_rotary_emb(xq, xk, freqs_cis)
        print(f'[RoPE] xq_rotate.shape: {xq_rotate.shape}')
        print(f'[RoPE] xk_rotate.shape: {xk_rotate.shape}')
        print(f'[RoPE] RoPETEST on device: {device} passed')

def RMSNormTEST(arg_map):
    rms_norm = RMSNorm(dim=arg_map.dim)
    for device in ['cpu', 'cuda:0']:
        x = torch.randn(
            (arg_map.max_batch_size, arg_map.max_seq_len, arg_map.dim),
            device=device
        )
        rms_norm.to(device)
        x_norm = rms_norm(x)
        print(f'[RMSNorm] x_norm.shape: {x_norm.shape}')
        print(f'[RMSNorm] RMSNormTEST on device: {device} passed')

def AttentionTEST(arg_map):
    for device in ['cpu', 'cuda:0']:
        head_dim = arg_map.dim // arg_map.n_heads
        x_norm = torch.randn(
            (arg_map.max_batch_size, arg_map.max_seq_len, arg_map.dim),
            device=device
        )
        xk = torch.randn(
            (arg_map.max_batch_size, arg_map.max_seq_len, arg_map.n_kv_heads, head_dim),
            device=device
        )
        n_rep = arg_map.n_heads // arg_map.n_kv_heads
        keys = Attention.repeat_kv(xk, n_rep)
        print(f'xk.shape: {xk.shape}')
        print(f'keys.shape: {keys.shape}')
        attention = Attention(arg_map)
        attention.to(device)
        x_out = attention(x_norm, start_pos=0)
        print(f'[Attention] x_out.shape: {x_out.shape}')
        print(f'[Attention] AttentionTEST on device: {device} passed')

def FeedForwardTEST(arg_map):
    for device in ['cpu', 'cuda:0']:
        x_out = torch.randn(
            (arg_map.max_batch_size, arg_map.max_seq_len, arg_map.dim),
            device=device
        )
        feed_forward = FeedForward(
            arg_map.dim, 4 * arg_map.dim, arg_map.multiple_of, arg_map.ffn_dim_multiplier
        )
        feed_forward.to(device)
        rms_norm = RMSNorm(dim=arg_map.dim)
        rms_norm.to(device)
        x_out = rms_norm(x_out)
        x_out = feed_forward(x_out)
        print(f'[FeedForward] x_out.shape: {x_out.shape}')
        print(f'[FeedForward] FeedForwardTEST on device: {device} passed')

def TransformerBlockTEST(arg_map):
    for device in ['cpu', 'cuda:0']:
        x = torch.randn((arg_map.max_batch_size, arg_map.max_seq_len, arg_map.dim), device=device)
        transformer_block = TransformerBlock(arg_map)
        transformer_block.to(device)
        transformer_block_out = transformer_block(x,start_pos=0)
        print(f'[TransformerBlock] transformer_block_out.shape: {transformer_block_out.shape}')
        print(f'[TransformerBlock] TransformerBlockTEST on device: {device} passed')

def TransformerTEST(arg_map):
    for device in ['cpu', 'cuda:0']:
        model = Transformer(arg_map).to(device)
        print('[Transformer] ', model)
        print(f'[Transformer] TransformerTEST on device: {device} passed')

# def DemoDataLoaderTEST(arg_map):
#     data_loader = DemoDataLoader('./data/demo/txt', arg_map.max_seq_len, arg_map.max_batch_size, 'full')
#     prompts = "Hello World"
#     encoded_tokens = data_loader.encode(prompts)
#     decoded_text = data_loader.decode(encoded_tokens)
#     print(f"Shakespeare text length: {data_loader.vocab_size}")
#     print(f"vocabulary content: {''.join(data_loader.vocab)}\n")
#     print(f"vocabulary size: {data_loader.vocab_size}")
#     print(f"encode result: {encoded_tokens}")
#     print(f"decode result: {decoded_text}")
#     print(f'[DemoDataLoader] DemoDataLoaderTEST passed')

def TokenizerTEST():
    tokenizer_dir = os.path.join('.', 'tokenizer')
    for tokenizer_file in os.listdir(tokenizer_dir):
        tokenizer = Tokenizer(os.path.join(tokenizer_dir, tokenizer_file))
        tokens = tokenizer.encode("This is a test sentence.", bos=True, eos=True)
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
    tokenizer_dir = os.path.join('.', 'tokenizer')
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


if __name__ == '__main__':
    arg_map = ModelArgs()
    RoPETEST(arg_map)
    RMSNormTEST(arg_map)
    AttentionTEST(arg_map)
    FeedForwardTEST(arg_map)
    TransformerBlockTEST(arg_map)
    TransformerTEST(arg_map)
    # DemoDataLoaderTEST(arg_map)
    TokenizerTEST()
    ChatFormatTEST()
    print('\n')
    print('==================== all tests passed ====================')
