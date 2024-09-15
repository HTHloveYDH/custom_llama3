import os
import json

import torch
import numpy as np
from data_pipeline.Tokenizer import Tokenizer, ChatFormat


class BaseSFTDataLoaderLite:
    def __init__(self, B, T, process_rank:int, num_processes:int, tokenizer_path:str, data_root:str, \
                 master_process:bool, split:str):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.tokenizer = Tokenizer(tokenizer_path)
        self.chat_format = ChatFormat(self.tokenizer)
        # get filenames
        files = os.listdir(data_root)  # all data files on current node
        split_files = [file for file in files if split in file]
        split_files = sorted(split_files)
        split_files = [os.path.join(data_root, file) for file in split_files]
        self.shards = split_files
        assert len(split_files) > 0, f'no shards found for split {split}'
        if master_process:
            print(f'found {len(split_files)} shards for split {split}')
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.data = self.load_data(self.shards[self.current_shard])
        self.current_position = self.B * self.process_rank
    
    def next_batch(self):
        B, T = self.B, self.T
        x, y = self.load_batch_tokens(self.data[self.current_position:self.current_position + B])  # get B samples for current GPU
        # advance the position in the tensor
        self.current_position += B * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * self.num_processes) > len(self.data):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.data = self.load_data(self.shards[self.current_shard])
            self.current_position = B * self.process_rank
        return x, y
    
    def load_data(self, filename:str):
        with open(filename, 'r') as f:
            json_content = json.load(f)
        # data format 1#: list = [[{'role': 'system', 'content': 'xxx'}, {'role': 'user', 'content': 'xxx'}, {'role': 'assistant', 'content': 'xxx'}], ...]
        # data format 2#: list = [{'instruction': 'xxx', 'input': 'xxx', 'output': 'xxx'}, ...]
        data = json_content['data']
        return data
    
    def load_batch_tokens(self, data:list):
        raise NotImplementedError(" Can not call 'load_batch_tokens' via base class 'BaseSFTDataLoaderLite'! ")

class InstructionSFTDataLoaderLite(BaseSFTDataLoaderLite):
    def __init__(self, B, T, process_rank:int, num_processes:int, tokenizer_path:str, data_root:str, \
                 master_process:bool, split:str):
        super(InstructionSFTDataLoaderLite, self).__init__(
            B, T, process_rank, num_processes, tokenizer_path, data_root, master_process, split
        )
    
    def load_batch_tokens(self, data:list):
        batch_prompt_tokens = []
        batch_output_tokens = []
        for dialog in data:
            # dialog: {'instruction': 'xxx', 'input': 'xxx', 'output': 'xxx'}
            dialog = [
                {
                    'role': 'system', 
                    'content': 'You are a chatbot, please be polite and imformative.'
                }, 
                {
                    'role': 'user', 
                    'content': self.complete_instruction(dialog['instruction'], dialog['input'])}, 
                {
                    'role': 'assistant', 
                    'content': dialog['output']
                }
            ]
            prompt_tokens = self.chat_format.encode_dialog_prompt(dialog[:-1], True, self.T)  # list
            batch_prompt_tokens.append(torch.tensor(prompt_tokens, dtype=torch.long))
            output_tokens = self.tokenizer.encode(
                dialog[-1]['content'], bos=True, eos=True, pad=True, max_len=self.T
            )
            batch_output_tokens.append(torch.tensor(output_tokens, dtype=torch.long))
        return torch.stack(batch_prompt_tokens, dim=0), torch.stack(batch_output_tokens, dim=0)
    
    def complete_instruction(self, instruction:str, context=None):
        completed_instruction = instruction
        if context:
            completed_instruction += '\n[CONTEXT]\n' + context
        return completed_instruction

class DialogSFTDataLoaderLite(BaseSFTDataLoaderLite):
    def __init__(self, B, T, process_rank:int, num_processes:int, tokenizer_path:str, data_root:str, \
                 master_process:bool, split:str):
        super(DialogSFTDataLoaderLite, self).__init__(
            B, T, process_rank, num_processes, tokenizer_path, data_root, master_process, split
        )
    
    def load_batch_tokens(self, data:list):
        batch_prompt_tokens = []
        batch_output_tokens = []
        for dialog in data:
            # dialog: [{'role': 'system', 'content': 'xxx'}, {'role': 'user', 'content': 'xxx'}, {'role': 'assistant', 'content': 'xxx'}]
            prompt_tokens = self.chat_format.encode_dialog_prompt(dialog[:-1], True, self.T)  # list
            batch_prompt_tokens.append(torch.tensor(prompt_tokens, dtype=torch.long))
            output_tokens = self.tokenizer.encode(
                dialog[-1]['content'], bos=True, eos=True, pad=True, max_len=self.T
            )
            batch_output_tokens.append(torch.tensor(output_tokens, dtype=torch.long))
        return torch.stack(batch_prompt_tokens, dim=0), torch.stack(batch_output_tokens, dim=0)
