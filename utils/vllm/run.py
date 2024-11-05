

import os

from utils.load_config import load_config_from_json
from utils.vllm.CustomLLM import CustomLLM


def main():
    llama_config, gen_config, cloud_config = load_config_from_json('gen')
    assert os.path.isdir(llama_config['ckpt_path'])
    assert os.path.isdir(llama_config['tokenizer_path'])
    # initialize custom LLM
    llm = CustomLLM(
        model_path=llama_config['ckpt_path'], 
        tokenizer_path=llama_config['tokenizer_path'], 
        params=llama_config['params']
    )
    # generate once
    if isinstance(prompt, str):
        prompt = gen_config['prompt']
        assert isinstance(prompt, str)
        response = llm.generate(prompt)
        print(f'Answer: {response}')
    # batch generation
    elif isinstance(prompt, list):
        prompts = gen_config['prompt']
        assert isinstance(prompts, list)
        for prompt in prompts:
            assert isinstance(prompt, str)
            response = llm.generate(prompt)
            print(f'\nQuestion: {prompt}')
            print(f'Answer: {response}')
    # generate multi-times (stream generation)
    elif isinstance(prompt, dict):
        prompt = gen_config['prompt']
        assert isinstance(prompt, str)
        for text in llm.generate(prompt, stream=True):
            print(text, end='', flush=True)
    else:
        raise ValueError('invalid prompt format')


if __name__ == '__main__':
    main()