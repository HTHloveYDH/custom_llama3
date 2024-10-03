import os
import sys
import time

import torch

sys.path.append(os.getcwd())
from dist.distribute import init_dist, ternimate_dist
# from data_pipeline.demo import DemoDataLoader
from data_pipeline.get_tokenizer import get_tokenizer
from data_pipeline.DataLoaderLiteFactory import DataLoaderLiteFactory
from models.get_model import get_model
from train.train_funcs import (
    st_train_on_epoch, st_valid_on_epoch, dpo_train_on_epoch, dpo_valid_on_epoch,
    resume_from_ckpt
)
# from gen.demo import generate
from gen.gen_funcs import generate
from utils.load_config import load_config_from_json as load_configs
from utils.logger import Logger


def main():
    ''' __________________________________________ setup _____________________________________________ '''
    # create the log directory we will write checkpoints to and log to
    log_dir = 'log'
    log_file_path = os.path.join('.', log_dir, f'log.log')
    # save log file
    sys.stdout = Logger(log_file_path, sys.stdout)
    # load configs
    llama3_config, train_config, data_config, cloud_config = load_configs('train')
    # llama3 configs
    dist = llama3_config['dist']
    dp, tp, pp = dist['dp'], dist['tp'], dist['pp']
    assert not (dist['dp_shard'] and dp == 1)
    assert not (dist['parallel_loss'] and tp == 1)
    assert not (dist['parallel_loss'] and dp > 1)
    tokenizer_path = llama3_config['tokenizer_path']
    lora = llama3_config['lora']
    # train configs
    training_type = train_config['training_type']
    assert training_type in ['pt', 'pre-train', 'sft', 'supervised-finetune', 'dpo', 'rlhf'], f'training type: {training_type} is not supported'
    dialog = training_type in ['sft', 'supervised-finetune']
    align = training_type in ['dpo', 'align']
    llama3_config['align'] = align
    learning_rate = train_config['learning_rate']  # defaults to 6e-4
    weight_decay = train_config['weight_decay']  # defaults to 0.1
    max_batch_size = train_config['max_batch_size']
    max_seq_len = train_config['max_seq_len']
    if llama3_config['model_type'] in ['llama3_8B', 'llama3_70B', 'llama3_405B']:
        assert max_batch_size == 32
        assert max_seq_len == 2048
    else:
        assert max_batch_size == llama3_config['params']['max_batch_size']
        assert max_seq_len == llama3_config['params']['max_seq_len']
    grad_accum_steps = train_config['grad_accum_steps']
    val_steps = train_config['val_steps']
    epochs = train_config['epochs']
    max_lr = train_config['max_lr']
    warmup_steps = train_config['warmup_steps']
    seed = train_config['seed']  # defaults to 1337
    gen_batch_size = train_config['gen_batch_size']
    assert gen_batch_size == max_batch_size
    gen_len = train_config['gen_len']
    temperature = train_config['temperature']
    top_p = train_config['top_p']
    prompt = train_config['prompt']
    log_interval = train_config['log_interval']
    ckpt_dir = train_config['ckpt_dir']
    # data configs
    data_root = data_config['data_root']
    data_format = data_config['data_format']
    total_token_num = data_config['total_token_num']
    # set up DP (distributed data parallel or fully sharded data parallel) process group.
    # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    master_process, device, device_mesh, parallel_args = init_dist(dist)
    # set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    assert total_token_num % (max_batch_size * max_seq_len * dp) == 0, 'make sure total_token_num is divisible by B * T * dp'
    steps_per_epoch = total_token_num // (max_batch_size * max_seq_len * dp)
    assert steps_per_epoch % grad_accum_steps == 0, 'make sure steps_per_epoch is divisible by grad_accum_steps'
    if master_process:
        print(f'total desired batch size: {total_token_num}')
        print(f'=> calculated gradient accumulation steps: {grad_accum_steps}')
    torch.set_float32_matmul_precision('high')

    ''' ________________________________________ load dataset ________________________________________ '''
    # train_data_loader = DemoDataLoader(data_root, max_seq_len, max_batch_size, 'train')
    # val_data_loader = DemoDataLoader(data_root, max_seq_len, max_batch_size, 'val')
    DataLoaderLite_factory = DataLoaderLiteFactory()
    # get global rank on data parallel level
    kwargs = {
        'B': max_batch_size, 'T': max_seq_len, 'process_rank': parallel_args.dp_global_rank,
        'num_processes': dp, 'tokenizer_path': tokenizer_path,
        'data_root': data_root, 'master_process': master_process, 'split': 'train'
    }
    train_data_loader = DataLoaderLite_factory.create(align, dialog, data_format, **kwargs)
    kwargs['split'] = 'val'
    val_data_loader = DataLoaderLite_factory.create(align, dialog, data_format, **kwargs)

    ''' ____________________________________ build & compile model ___________________________________ '''
    kwargs = {'learning_rate': learning_rate, 'weight_decay': weight_decay}
    model, optimizer = get_model(llama3_config, device_mesh, device, True, **kwargs)

    ''' ____________________________________________ train ___________________________________________ '''
    # get tokenizer
    tokenizer, chat_format = get_tokenizer(tokenizer_path)
    # get functions
    train_on_epoch, valid_on_epoch = st_train_on_epoch, st_valid_on_epoch  # supervised training
    if align:
        train_on_epoch, valid_on_epoch = dpo_train_on_epoch, dpo_valid_on_epoch  # dpo training
    # start train loop
    resume_from_ckpt(model, ckpt_dir)
    for epoch in range(epochs):
        print(f'epoch: {epoch} / {epochs}:')
        # train llm for one epoch
        train_on_epoch(
            model, train_data_loader, optimizer, device, steps_per_epoch, grad_accum_steps, epoch, 
            log_interval, parallel_args, master_process
        )
        # validate current weights on validation dataset shard of current process
        valid_on_epoch(
            model, val_data_loader, device, val_steps, epoch, parallel_args, master_process, lora
        )
        # generate sentences to verify current weights in the master process
        if master_process and not tp > 1:
            # _, _ = generate(
            #     model, "Hello, I'm a language model,", gen_batch_size, gen_len, temperature, top_p,
            #     device
            # )
            return_messages = generate(
                model, tokenizer, chat_format, prompt, device, gen_batch_size, gen_len, dialog,
                parallel_args.dp_global_rank
            )
    # terminate process group
    ternimate_dist(dist)


if __name__ == '__main__':
    main()
