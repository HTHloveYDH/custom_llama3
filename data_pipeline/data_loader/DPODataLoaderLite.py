import json

import torch

from data_pipeline.data_loader.BaseDataLoaderLite import BaseDataLoaderLite


class BaseDPODataLoaderLite(BaseDataLoaderLite):
    def __init__(self, B, T, process_rank:int, num_processes:int, tokenizer_path:str, data_root:str, \
                 master_process:bool, split:str):
        super(BaseDPODataLoaderLite, self).__init__(
            B, T, process_rank, num_processes, tokenizer_path, data_root, master_process, split
        )
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
        # data format 2#: list = [{'prompt': 'xxx', 'winner_response': 'xxx', 'loser_response': 'xxx'}, ...]
        data = json_content['data']
        return data
    
    def load_batch_tokens(self, data:list):
        batch_winner_prompt_tokens = []
        batch_loser_output_tokens = []
        for dialog in data:
            # dialog: {'prompt': 'xxx', 'winner_response': 'xxx', 'loser_response': 'xxx'}
            winner_prompt_tokens = self.tokenizer.encode(
                dialog['winner_response'], bos=True, eos=True, pad=True, max_len=self.T
            )  # list
            loser_output_tokens = self.tokenizer.encode(
                dialog['loser_response'], bos=True, eos=True, pad=True, max_len=self.T
            )  # list
            dialog = [
                {
                    'role': 'system', 
                    'content': 'You are a chatbot, please be polite and imformative.'
                }, 
                {
                    'role': 'user', 
                    'content': dialog['prompt']
                }
            ]
            prompt_tokens = self.chat_format.encode_dialog_prompt(dialog, True, self.T)  # list
            winner_prompt_tokens = prompt_tokens + winner_prompt_tokens
            loser_output_tokens = prompt_tokens + loser_output_tokens
            batch_winner_prompt_tokens.append(torch.tensor(winner_prompt_tokens, dtype=torch.long))
            batch_loser_output_tokens.append(torch.tensor(loser_output_tokens, dtype=torch.long))
        return torch.stack(batch_winner_prompt_tokens, dim=0), torch.stack(batch_loser_output_tokens, dim=0)
