## lanuch training task:
```bash
# [reference]: https://www.youtube.com/watch?v=KaAJtI1T2x4&list=PL_lsbAsL_o2CSuhUhJIiW0IkdT5C2wGWj
```
### simple launch on one node:
```bash
python train/main.py --dp_world_size 1
```

### DDP (FSDP) launch on one node by torch.multiprocessing (e.g. 8 GPUs):
```bash
python train/main.py --dp_world_size 8 --torch_mp_launch
```

### DDP (FSDP) launch on one node by torchrun (e.g. 8 GPUs):
```bash
torchrun --standalone --nproc_per_node=8 train/main.py --dp_world_size 8
```

### DDP (FSDP) launch on multi node by torchrun (e.g. 2 * 8 GPUs, two nodes):
```bash
# on node 0#
torchrun --nproc_per_node=8 --nnodes=2 -node_rank=0 --rdzv_backend=c10d --rdzv_endpoint=xxx.xxx.xxx.xxx:xxxx train/main.py --dp_world_size 16
```

```bash
# on node 1#
torchrun --nproc_per_node=8 --nnodes=2 -node_rank=1 --rdzv_backend=c10d --rdzv_endpoint=xxx.xxx.xxx.xxx:xxxx train/main.py --dp_world_size 16
```

## lanuch generation task:
```bash
python gen/main.py
```

## lanuch test code:
```bash
python test/module_tests.py
```

## llama3 configs:
### llama3_8B
    dim=4096, n_layers=32, n_heads=32, n_kv_heads=8, vocab_size=128256, 
    multiple_of=1024, ffn_dim_multiplier=1.3, norm_eps=1e-5, rope_theta=500000.0, 
    max_batch_size=32, max_seq_len=2048
### llama3_70B
    dim=8192, n_layers=80, n_heads=64, n_kv_heads=8, vocab_size=128256, 
    multiple_of=4096, ffn_dim_multiplier=1.3, norm_eps=1e-5, rope_theta=500000.0, 
    max_batch_size=32, max_seq_len=2048
### llama3_405B
    dim=16384, n_layers=126, n_heads=128, n_kv_heads=8, vocab_size=128256, 
    multiple_of=None, ffn_dim_multiplier=1.3, norm_eps=1e-5, rope_theta=500000.0, 
    max_batch_size=32, max_seq_len=2048

## env configuration:
### env pytorch
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install transformers==4.44.0

pip install tiktoken==0.7.0

pip install tqdm==4.66.5
```

### tensorrt-llm
```bash
cd TensorRT-LLM/examples/bloom

pip install torch torchvision torchaudio (2.4.0, cuda 12.1)

conda install -y mpi4py

conda install openmpi

pip install tensorrt_llm==0.13.0.dev2024081300 --extra-index-
url https://pypi.nvidia.com

pip install -r ./requirements.txt
```

## TODO:
1. add AWS s3 support to let several nodes read from AWS S3 bucket during multi-node training task

## some useful links:
### quant
https://chatgpt.com/share/31aa8af3-dce2-457f-85db-2b18b3c242ce
