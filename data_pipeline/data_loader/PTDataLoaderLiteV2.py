import json

import torch
import numpy as np

from data_pipeline.data_loader.BaseDataLoaderLite import BaseDataLoaderLite


class BasePTDataLoaderLiteV2(BaseDataLoaderLite):
    def __init__(self, B, T, process_rank:int, num_processes:int, tokenizer_path:str, data_root:str, \
                 master_process:bool, split:str):
        super(BasePTDataLoaderLiteV2, self).__init__(
            B, T, process_rank, num_processes, tokenizer_path, data_root, master_process, split
        )
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.data = self.load_data(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
    
    def next_batch(self):
        B, T = self.B, self.T
        x, y = self.load_batch_tokens(self.data[self.current_position:self.current_position + B])  # get B samples for current GPU
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self.load_data(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y
    
    def load_data(self, filename:str):
        raise NotImplementedError(" Can not call 'load_data' via base class 'BasePTDataLoaderLiteV2'! ")
    
    def load_batch_tokens(self, data:list):
        batch_x_tokens = []
        batch_y_tokens = []
        for text in data:
            tokens = self.tokenizer.encode(text, bos=True, eos=True, pad=True, max_len=self.T + 1)
            batch_x_tokens.append(torch.tensor(tokens[:-1], dtype=torch.long))
            batch_y_tokens.append(torch.tensor(tokens[1:], dtype=torch.long))
        return torch.stack(batch_x_tokens, dim=0), torch.stack(batch_y_tokens, dim=0)

class NpyPTDataLoaderLiteV2(BasePTDataLoaderLiteV2):
    def __init__(self, B, T, process_rank:int, num_processes:int, tokenizer_path:str, data_root:str, \
                 master_process:bool, split:str):
        super(NpyPTDataLoaderLiteV2, self).__init__(
            B, T, process_rank, num_processes, tokenizer_path, data_root, master_process, split
        )
    
    def load_data(self, filename:str):
        np_content = np.load(filename)  # np.array([1, 2, 3], ...)
        # data format: list = [[1, 2, 3], ...]
        data = [x for x in np_content]
        return data
    
    def load_batch_tokens(self, data:list):
        batch_x_tokens = []
        batch_y_tokens = []
        for tokens in data:
            batch_x_tokens.append(torch.tensor(tokens, dtype=torch.long))
            batch_y_tokens.append(torch.tensor(tokens, dtype=torch.long))
        return torch.stack(batch_x_tokens, dim=0), torch.stack(batch_y_tokens, dim=0)

class TxtPTDataLoaderLiteV2(BasePTDataLoaderLiteV2):
    def __init__(self, B, T, process_rank:int, num_processes:int, tokenizer_path:str, data_root:str, \
                 master_process:bool, split:str):
        super(TxtPTDataLoaderLiteV2, self).__init__(
            B, T, process_rank, num_processes, tokenizer_path, data_root, master_process, split
        )

    def load_data(self, filename:str):
        with open(filename, 'r') as f:
            text_lines = f.readlines()  # ['xxx', ...]
        # data format: list = ['xxx', ...]
        data = [x.strip() for x in text_lines]
        return data

class JsonPTDataLoaderLiteV2(BasePTDataLoaderLiteV2):
    def __init__(self, B, T, process_rank:int, num_processes:int, tokenizer_path:str, data_root:str, \
                 master_process:bool, split:str):
        super(TxtPTDataLoaderLiteV2, self).__init__(
            B, T, process_rank, num_processes, tokenizer_path, data_root, master_process, split
        )

    def load_data(self, filename:str):
        with open(filename, 'r') as f:
            json_content = json.load(f)  # {'data': [{'text': 'xxx'}, ...]}
        # data format: list = ['xxx', ...]
        data = [x['text'] for x in json_content['data']]
        return data

