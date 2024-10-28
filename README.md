## lanuch training task:

```bash
# [reference]: https://www.youtube.com/watch?v=KaAJtI1T2x4&list=PL_lsbAsL_o2CSuhUhJIiW0IkdT5C2wGWj
```

### simple launch on one node:

```bash
python train/main.py
```

### DDP (FSDP) launch on one node by torchrun (e.g. 8 GPUs):

```bash
torchrun --standalone --nproc_per_node=8 train/main.py
```

### DDP (FSDP) launch on multi node by torchrun (e.g. 2 * 8 GPUs, two nodes):

```bash
# on node 0#
torchrun --nproc_per_node=8 --nnodes=2 -node_rank=0 --rdzv_backend=c10d --rdzv_endpoint=xxx.xxx.xxx.xxx:xxxx train/main.py
```

```bash
# on node 1#
torchrun --nproc_per_node=8 --nnodes=2 -node_rank=1 --rdzv_backend=c10d --rdzv_endpoint=xxx.xxx.xxx.xxx:xxxx train/main.py
```

## lanuch generation task:

```bash
python gen/main.py
```

## lanuch generation task with tensor parallelism:

```bash
torchrun --standalone --nproc_per_node=2 gen/main.py
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

### base pytorch env

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install transformers==4.44.0

pip install tiktoken==0.7.0

pip install blobfile==3.0.0

pip install tqdm==4.66.5
```

### faiss env

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install faiss-cpu==1.8.0.post1

pip install transformers==4.44.0

pip install tiktoken==0.7.0

pip install blobfile==3.0.0

pip install tqdm==4.66.5
```

### fairscale env

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install fairscale==0.4.13

pip install fire==0.7.0

pip install transformers==4.44.0

pip install tiktoken==0.7.0

pip install blobfile==3.0.0

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

# Quant Things

## Pre  Doings

`Depend On Docker Env `

1. Prepare source code

   ```shell
   git lfs install   # Open git large file transport
   git clone https://github.com/NVIDIA/TensorRT-LLM.git
   git checkout v0.8.0  #change version to 0.8.0
   git submodule update --init --recursive  # update some sumbodules
   git lsf pull
   ```
2. build docker image

   ```shell
   make -C docker release_build
   ```

   and you will get a docker image
3. run docker container

   run from scratch

   ```shell
   make -C docker release_run
   ```

   mapping the file is better

   ```
   makr -C docker release_run DOCKER_RUN_ARGS="-v  /host/machine/path/:/container/path"
   ```

   Tips: Run this

   cd This path

   then, do some about quant things 👇，good luck😀

cd  /app/tensorrt_llm/examples/llama

## Intro

`Quantization happened in generate checkpoint stage.  Beside convert_checkpoint.py, another script is quantize.py, This two scripts include different quant functions. Here is a demo, shows different quantization methods, LLaMA 7B based.`

## Int8 weight-only

```shell
python convert_checkpoint.py --model_dir /home/Llama-2-7b-hf/  --output_dir ./checkpoint_1gpu_int8_wo --dtype float16 --use_weight_only --weight_only_precision int8
```

## Int4 wight-only

```shell
python convert_checkpoint.py --model_dir /home/Llama-2-7b-hf/  --output_dir ./checkpoint_1gpu_int8_wo --dtype float16 --use_weight_only --weight_only_precision int4
```

## Int4 AWQ

```
python ../quantization/quantize.py --model_dir /home/Llama-2-7b-hf/  --output_dir ./checkpoint_1gpu_int8_wo --dtype float16 --qformat int4_awq --awq_block_size 128 --calib_size 32
```

## Int4 GPTQ

use part of full model  `model-00001-of-00002.safetensors`

```shell
python convert_checkpoint.py --model_dir /home/Llama-2-7b-hf/  --output_dir ./checkpoint_1gpu_int8_wo --dtype float16 --use_weight_only --weight_only_precision int4_gptq --per_group --ammo_quant_ckpt_path ./llama-7b-4bit-gs128.safetensors
```

## Int8 SmoothQuant

This quant method also in convert_checkpoint.py, use the `--smoothquant` parameter to open.

also use `--per_token` & `--per_channel` , 0.5 means transfer 50% activation quant stress to weight quant, now let's us consider the corner case, if 0, that mean none-transfer the quant stress, it's hard to quant activation part, easy to quant weight part. if 1,that mean transfer all stress to weight part,  cause easy to quant activate part, hard to quant weight part.

```python
python convert_checkpoint.py --model_dir /home/Llama-2-7b-hf/  --output_dir ./checkpoint_1gpu_int8_wo --dtype float16 --smoothquant 0.5 --per_token --per_channel
```

Tips: if you meet `trust_remote_code` problem,  just run  `export HF_DATASETS_TRUST_REMOTE_CODE=1` in your terminal, it will download `cnn_stories.tgz` and `dailymail_stories.tgz.`

## FP8

`--calib_size 512` Optional Parameter

```shell
python ../quantization/quantize.py --model_dir /home/Llama-2-7b-hf/  --output_dir ./checkpoint_1gpu_int8_wo --dtype float16 --qformat fp8 --calib_size 512
```

## INT8 KV_Cache

```shell
python convert_checkpoint.py --model_dir /home/Llama-2-7b-hf/  --output_dir ./checkpoint_1gpu_int8_wo --dtype float16 --int8_kv_cache
```

👆 file type .safetensors to .safetensors

👇 file type .safetensors to .engine

## Engine Generate

After quant, safetensors to engine

```shell
trtllm-build --checkpoint_dir ./checkpoint_1gpu_int8_wo --output_dir ./engine_1gpu_int8_wo --gemm_plugin float16 --max_batch_size 16 --max_input_len 2048 --max_output_len 2048
```

If you meet this problem `ValueError: max_workers must be greater than 0`, need to mapping CUDA in Container.

```shell
sudo make -C docker release_run DOCKER_RUN_ARGS=" --gpus all -v /data_ws/Data_1/tinghao/boxue/llm_weight:/home"
```

and you see  gpus now,

the get the  engine😀

## Outro

Actually，lots of quant methods in [TensorRT-LLM/examples/llama at main · NVIDIA/TensorRT-LLM (github.com)](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama).  Can test them one by one.

## LLama Like Model All Steps

## `    ~ From begin to end,from train to quant ~`

## Intro

All files in myllama, path in docker `/home/myllama`

First of all, generate and organize files, `LLama` file type like !   :)

Include `config.json,`

`model-00001-of-00004.safetensors,`

`model-00002-of-00004.safetensors,`

`model-00003-of-00004.safetensors,`

`model-00004-of-00004.safetensors,`

`pytorch_model-00001-of-00004.bin,`

`pytorch_model-00002-of-00004.bin,`

`pytorch_model-00003-of-00004.bin,`

`pytorch_model-00004-of-00004.bin,`

`model.safetensors.index.json`

## fp16  To int8

```python
python convert_checkpoint.py --model_dir /home/Llama-2-7b-hf/ --output_dir ./myllama_1gpu_int8_wo --dtype float16 --use_weight_only --weight_only_precision int8
```

`skip the type check function :)`

## safetensors To engine

```shell
trtllm-build --checkpoint_dir ./myllama_1gpu_int8_wo --output_dir ./myllama_1gpu_int8_wo_engine --gemm_plugin float16 --max_batch_size 16 --max_input_len 2048 --max_output_len 2048
```

## Inference

```python
 python3 run.py --engine_dir ./llama/myllama_1gpu
_int8_wo_engine/ --max_output_len 100 --tokenizer_dir /home/Llama-2-7b-hf/ --input_text "How do I count to nine in French?"
```

## Ref

1. [TensorRT-LLM/examples/llama at main · NVIDIA/TensorRT-LLM (github.com)](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama)
2. [Installing the NVIDIA Container Toolkit — NVIDIA Container Toolkit 1.16.2 documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## TODO:

1. add AWS s3 support to let several nodes read data from AWS S3 bucket during multi-node training task

## some useful links:

### quant

https://chatgpt.com/share/31aa8af3-dce2-457f-85db-2b18b3c242ce

### torch.distributed

https://pytorch.org/docs/stable/distributed.html
The package (torch.distributed) needs to be initialized using the torch.distributed.init_process_group() or torch.distributed.device_mesh.init_device_mesh() function before calling any other methods. Both block until all processes have joined.

### torch.distributed.tensor.parallel

https://pytorch.org/docs/stable/distributed.tensor.parallel.html

### Pytorch: Pipeline Parallelism

https://pytorch.org/docs/stable/distributed.pipelining.html
https://pytorch.org/tutorials/intermediate/pipelining_tutorial.html

### huggingface safetensors to llama.cpp gguf format

https://huggingface.co/docs/transformers/main/en/gguf

### useful links about Transformer

https://huggingface.co/docs/transformers/main/model_doc/llama2
https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L255

https://huggingface.co/docs/transformers/main/model_doc/llama3
https://github.com/meta-llama/llama3

https://spaces.ac.cn/archives/9708
https://spaces.ac.cn/archives/9948

https://github.com/bojone/rerope?tab=readme-ov-file

https://github.com/hkproj/pytorch-lora

https://github.com/pytorch/torchtitan/tree/main

https://huggingface.co/docs/transformers/main/en/gguf

https://pytorch.org/docs/stable/distributed.pipelining.html
https://pytorch.org/tutorials/intermediate/pipelining_tutorial.html
