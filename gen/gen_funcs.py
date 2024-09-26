import time
import json

import torch
from torch.nn import functional as F

from gen.rag import query_database
from utils.get_device_type import get_device_type

def generate_tokens(model, tokens:list, gen_batch_size:int, gen_len:int, device:str, \
                    dp_global_rank:int):
    assert isinstance(tokens, list)
    device_type = get_device_type(device)
    tokens = torch.tensor(tokens, dtype=torch.long)  # shape: (len(prompt))
    tokens = tokens.unsqueeze(0).repeat(gen_batch_size, 1)  # shape: (gen_batch_size, len(prompt))
    xgen = tokens.to(device)  # current shape: [gen_batch_size, len(prompt))
    xcol = xgen
    # sampling configuration
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42 + dp_global_rank)
    start_pos = 0
    while xgen.size(1) < gen_len:
        # forward the model to get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits = model(xcol, start_pos)  # (B, T, vocab_size)
            start_pos += xcol.size(1)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            xgen = torch.cat((xgen, xcol), dim=1)
    return xgen

def generate(model, tokenizer, chat_format, prompt, device:str, gen_batch_size:int, \
             gen_len:int, dialog:bool, dp_global_rank=0):
    model.eval()
    # preprocess for input prompt: python <class 'list'>
    if dialog:
        assert isinstance(prompt, list)  # example: [{'role': 'system', 'content': 'xxx.',}, {'role': 'user', 'content': 'xxx.',}]
        tokens = chat_format.encode_dialog_prompt(prompt)  # python <class 'list'>
    else:
        assert isinstance(prompt, str)  # 'Hello, I am a student.'
        tokens = tokenizer.encode(prompt, bos=True, eos=True)  # python <class 'list'>
    xgen = generate_tokens(model, tokens, gen_batch_size, gen_len, device, dp_global_rank)
    return_messages = []
    # print the generated text
    for i in range(gen_batch_size):
        tokens = xgen[i, :gen_len].tolist()
        decoded = tokenizer.decode(tokens)
        print(f'[generation text] rank {dp_global_rank} sample {i}: {decoded}')
        if dialog:
            return_messages.append({'generation': {'role': 'assistant', 'content': decoded}})
        else:
            return_messages.append({'generation': decoded})
    return return_messages

'''reference from https://github.com/bklieger-groq/g1/blob/main/g1.py'''
def get_model_response(model, cot_format, tokenizer, cot_prompt, gen_len:int, is_final_answer:bool, \
                       device:str, dp_global_rank:int):
    tokens = cot_format.encode_dialog_prompt(cot_prompt)  # python <class 'list'>
    # sampling configuration
    for attempt in range(3):
        try:
            xgen = generate_tokens(model, tokens, 1, gen_len, device, dp_global_rank)
            assert xgen.size(0) == 1
            tokens = xgen[0, :gen_len].tolist()
            response = tokenizer.decode(tokens)
            if is_final_answer:
                return response
            else:
                return json.loads(response)
        except Exception as e:
            if attempt == 2:
                if is_final_answer:
                    return {'title': 'Error', 'content': f'Failed to generate final answer after 3 attempts. Error: {str(e)}'}
                else:
                    return {'title': 'Error', 'content': f'Failed to generate step after 3 attempts. Error: {str(e)}', 'next_action': 'final_answer'}
            time.sleep(1)  # Wait for 1 second before retrying

def cot_generate(model, tokenizer, cot_format, prompt:str, device:str, gen_len:int, \
                 dp_global_rank=0):
    model.eval()
    cot_prompt = [
        {'role': 'system', 'content': '''You are an expert AI assistant that explains your reasoning step by step. For each step, provide a title that describes what you're doing in that step, along with the content. Decide if you need another step or if you're ready to give the final answer. Respond in JSON format with 'title', 'content', and 'next_action' (either 'continue' or 'final_answer') keys. USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES.

Example of a valid JSON response:
```json
{
    'title': 'Identifying Key Information',
    'content': 'To begin solving this problem, we need to carefully examine the given information and identify the crucial elements that will guide our solution process. This involves...',
    'next_action': 'continue'
}```
'''},
        {'role': 'user', 'content': prompt},
        {'role': 'assistant', 'content': 'Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem.'}
    ]
    steps = []
    step_count = 1
    total_think_time = 0
    while True:
        start_time = time.time()
        step_data = get_model_response(
            model, cot_format, tokenizer, cot_prompt, gen_len // 4, False, device, dp_global_rank
        )  # json format
        end_time = time.time()
        think_time = end_time - start_time
        total_think_time += think_time
        steps.append((f"Step {step_count}: {step_data['title']}", step_data['content'], think_time))
        cot_prompt.append({'role': 'assistant', 'content': json.dumps(step_data)})
        if step_data['next_action'] == 'final_answer' or step_count > 25: # Maximum of 25 steps to prevent infinite think time. Can be adjusted.
            break
        step_count += 1
        # Yield after each step for Streamlit to update
        # yield steps, None  # We're not yielding the total time until the end
        print(f'[cot generation text] rank {dp_global_rank}: {steps}')
    # Generate final answer
    cot_prompt.append({'role': 'user', 'content': 'Please provide the final answer based solely on your reasoning above. Do not use JSON formatting. Only provide the text response without any titles or preambles. Retain any formatting as instructed by the original prompt, such as exact formatting for free response or multiple choice.'})
    start_time = time.time()
    final_data = get_model_response(
        model, cot_format, tokenizer, cot_prompt, gen_len, True, device, dp_global_rank
    )  # text format, not json format
    end_time = time.time()
    think_time = end_time - start_time
    total_think_time += think_time
    steps.append(('Final Answer', final_data, think_time))
    # yield steps, total_think_time
    print(f'[cot final generation text] rank {dp_global_rank}: {steps}')
    return steps, total_think_time

def rag_generate(model, tokenizer, rag_format, prompt:str, device:str, gen_len:int, \
                 dialog:bool, database_path:str, raw_txt_data_path:str, dp_global_rank=0):
    # retrive from database
    return_info_list = query_database(prompt, database_path, raw_txt_data_path, 3, verbose=False)
    if dialog:
        rag_prompt = [
            {'role': 'system', 'content': 'You are an expert AI assistant. Please be polite and informative.'},
            {'role': 'user', 'content': prompt + '\n[CONTEXT]\n'}
        ]
        for info in return_info_list:
            rag_prompt[1]['content'] += info + '\n'
    else:
        rag_prompt = prompt + '\n[CONTEXT]\n'
        for info in return_info_list:
            rag_prompt += info + '\n'
    return_messages = generate(
        model, tokenizer, rag_format, rag_prompt, device, 1, gen_len, dialog, dp_global_rank
    )
    return return_messages
