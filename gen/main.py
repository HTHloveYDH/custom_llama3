import os
import sys
import time

import torch

sys.path.append(os.getcwd())
from dist.distribute import init_dist, ternimate_dist
from data_pipeline.get_tokenizer import get_tokenizer
from models.get_model import get_model
# from gen.demo import generate
from gen.gen_funcs import generate
from utils.load_config import load_config_from_json as load_configs


def main():
    ''' __________________________________________ setup _____________________________________________ '''
    llama3_config, gen_config, cloud_config, dist_config = load_configs('gen')
    # distribute configs
    dist_strategy = dist_config['dist_strategy']
    assert dist_strategy in ['ddp', 'fsdp', 'default'], f'distribute strategy: {dist_strategy} is not supported'
    # generation configs
    dialog = gen_config['dialog']
    seed = gen_config['seed']  # defaults to 1337
    gen_batch_size = gen_config['gen_batch_size']
    gen_len = gen_config['gen_len']
    if llama3_config['model_type'] in ['llama3_8B', 'llama3_70B', 'llama3_405B']:
        assert gen_batch_size == 2048
    else:
        assert gen_batch_size == llama3_config['params']['max_batch_size']
    temperature = gen_config['temperature']
    top_p = gen_config['top_p']
    prompt = gen_config['prompt']
    # llama3 configs
    tokenizer_path = llama3_config['tokenizer_path']
    use_compile = llama3_config['use_compile']
    # set up DP (distributed data parallel or fully sharded data parallel).
    # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    dp = dist_strategy in ['ddp', 'fsdp']
    dp_global_rank, dp_local_rank, dp_world_size, master_process, device, _ = init_dist(
        dist_strategy, False, 0, 1
    )
    # set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_float32_matmul_precision('high')

    ''' ____________________________________ build & compile model ___________________________________ '''
    device_ids = [dp_local_rank]
    model, raw_model = get_model(llama3_config, device, dist_strategy, device_ids)

    ''' ____________________________________________ test ___________________________________________ '''
    # _, _ = generate(model, prompt, gen_batch_size, gen_len, temperature, top_p, device=device)
    # get tokenizer
    tokenizer, chat_format = get_tokenizer(tokenizer_path)
    generate(
        model, tokenizer, chat_format, prompt, device, gen_batch_size, gen_len, dialog, 
        dp_global_rank
    )
    ternimate_dist(dist_strategy)


if __name__ == '__main__':
    main()