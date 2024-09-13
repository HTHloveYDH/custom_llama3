import os
import json

import torch
import numpy as np
from data_pipeline.Tokenizer import Tokenizer, ChatFormat


class SFTDataLoaderLite:
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
        self.dialogs = self.load_dialogs(self.shards[self.current_shard])
        self.current_position = self.B * self.process_rank
    
    def next_batch(self):
        B, T = self.B, self.T
        x, y = self.load_batch_tokens(self.dialogs[self.current_position:self.current_position + B + 1])
        # advance the position in the tensor
        self.current_position += B * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * self.num_processes + 1) > len(self.dialogs):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.dialogs = self.load_dialogs(self.shards[self.current_shard])
            self.current_position = B * self.process_rank
        return x, y
    
    def load_dialogs(self, filename:str):
        with open(filename, 'r') as f:
            json_content = json.load(f)
        dialogs = json_content['dialogs']  # list: [[{}, {}, {}], ...]
        return dialogs
    
    def load_batch_tokens(self, dialogs:list):
        batch_prompt_tokens = []
        batch_output_tokens = []
        for dialog in dialogs:
            prompt_tokens = self.chat_format.encode_dialog_prompt(dialog[:-1], True, self.T)  # list
            batch_prompt_tokens.append(torch.tensor(prompt_tokens, dtype=torch.long))
            output_tokens = self.tokenizer.encode(
                dialog[:-1]['assistant'], bos=True, eos=True, pad=True, max_len=self.T
            )
            batch_output_tokens.append(torch.tensor(output_tokens, dtype=torch.long))
        return torch.stack(batch_prompt_tokens, dim=0), torch.stack(batch_output_tokens, dim=0)
