import os
import sys
import time

import torch
import statistics
sys.path.append(os.getcwd())
from dist.distribute import init_dist, ternimate_dist
from data_pipeline.get_tokenizer import get_tokenizer
from models.get_model import get_model
# from gen.demo import generate
from gen.gen_funcs import generate, cot_generate
from utils.load_config import load_config_from_json as load_configs


def main():
    ''' __________________________________________ setup _____________________________________________ '''
    llama3_config, gen_config, cloud_config, dist_config = load_configs('gen')
    # distribute configs
    dist_type = dist_config['dist_type']
    assert dist_type in ['tp', 'default'], f'distribute strategy: {dist_type} is not supported'
    dp_size = dist_config['data_parallel_size']
    tp_size = dist_config['tensor_parallel_size']
    # generation configs
    dialog = gen_config['dialog']
    cot = gen_config['cot']
    seed = gen_config['seed']  # defaults to 1337
    gen_batch_size = gen_config['gen_batch_size']
    gen_len = gen_config['gen_len']
    if llama3_config['model_type'] in ['llama3_8B', 'llama3_70B', 'llama3_405B']:
        assert gen_batch_size == 32
    else:
        assert gen_batch_size == llama3_config['params']['max_batch_size']
    temperature = gen_config['temperature']
    top_p = gen_config['top_p']
    prompt = gen_config['prompt']
    # llama3 configs
    tokenizer_path = llama3_config['tokenizer_path']
    use_compile = llama3_config['use_compile']
    llama3_config['align'] = False
    # set up DP (distributed data parallel or fully sharded data parallel).
    # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    dp_global_rank, dp_local_rank, device_mesh, master_process, device, _ = init_dist(
        dist_type, dp_size, tp_size, False, 0
    )
    # set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_float32_matmul_precision('high')

    ''' ____________________________________ build & compile model ___________________________________ '''
    model, raw_model = get_model(llama3_config, device, dist_type, device_mesh)

    ''' ____________________________________________ test ___________________________________________ '''
    # _, _ = generate(model, prompt, gen_batch_size, gen_len, temperature, top_p, device=device)
    # get tokenizer
    tokenizer, chat_format = get_tokenizer(tokenizer_path)
    
    # Prepare for timing
    generation_times = []
    num_runs = 5  # Number of times to run the generation for statistical analysis

    for _ in range(num_runs):
        start_time = time.time()
        
        if cot:
            steps, think_time = cot_generate(
                model, tokenizer, chat_format, prompt, device, gen_len, dp_global_rank
            )
        else:
            return_messages = generate(
                model, tokenizer, chat_format, prompt, device, gen_batch_size, gen_len, dialog, 
                dp_global_rank
            )
        
        end_time = time.time()
        generation_time = end_time - start_time
        generation_times.append(generation_time)

    # Calculate statistics
    avg_time = statistics.mean(generation_times)
    std_dev = statistics.stdev(generation_times) if len(generation_times) > 1 else 0
    min_time = min(generation_times)
    max_time = max(generation_times)

    # Print results
    if master_process:
        print(f"Generation Statistics (over {num_runs} runs):")
        print(f"Average Time: {avg_time:.4f} seconds")
        print(f"Standard Deviation: {std_dev:.4f} seconds")
        print(f"Minimum Time: {min_time:.4f} seconds")
        print(f"Maximum Time: {max_time:.4f} seconds")

    ternimate_dist(dist_type)


if __name__ == '__main__':
    main()
