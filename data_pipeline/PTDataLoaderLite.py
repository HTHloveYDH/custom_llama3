import os
import json

import torch
import numpy as np
from data_pipeline.Tokenizer import Tokenizer


class BasePTDataLoaderLite:
    def __init__(self, B, T, process_rank:int, num_processes:int, tokenizer_path:str, data_root:str, \
                 master_process:bool, split:str):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.tokenizer = Tokenizer(tokenizer_path)
        assert split in {'train', 'val'}
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
        self.tokens = self.load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
    
    def next_batch(self):
        B, T = self.B, self.T
        buffer = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = (buffer[:-1]).view(B, T)  # inputs
        y = (buffer[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self.load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y
    
    def load_tokens(self, filename:str):
        NotImplementedError(" Can not call 'load_tokens' via base class 'BasePTDataLoaderLite'! ")

class NpyPTDataLoaderLite(BasePTDataLoaderLite):
    def __init__(self, B, T, process_rank:int, num_processes:int, tokenizer_path:str, data_root:str, \
                 master_process:bool, split:str):
        super(NpyPTDataLoaderLite, self).__init__(
            B, T, process_rank, num_processes, tokenizer_path, data_root, master_process, split
        )
    
    def load_tokens(self, filename:str):
        np_tokens = np.load(filename)
        np_tokens = np_tokens.astype(np.int32) # added after video
        tensor_tokens = torch.tensor(np_tokens, dtype=torch.long)
        return tensor_tokens

class TxtPTDataLoaderLite(BasePTDataLoaderLite):
    def __init__(self, B, T, process_rank:int, num_processes:int, tokenizer_path:str, data_root:str, \
                 master_process:bool, split:str):
        super(TxtPTDataLoaderLite, self).__init__(
            B, T, process_rank, num_processes, tokenizer_path, data_root, master_process, split
        )

    def load_tokens(self, filename:str):
        with open(filename, 'r') as f:
            text = f.read()
        tokens = self.tokenizer.encode(text, bos=True, eos=True)
        tensor_tokens = torch.tensor(tokens, dtype=torch.long)
        return tensor_tokens
