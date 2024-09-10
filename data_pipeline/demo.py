import os

import torch


class DemoDataLoader:
    def __init__(self, data_root:str, seq_len:int, batch_size:int, split:str):
        with open(os.path.join(data_root, 'tiny_shakespeare.txt'), 'r') as f:  
            data = f.read()
        vocab = sorted(list(set(data)))
        vocab.extend(['<|begin_of_text|>','<|end_of_text|>','<|pad_id|>'])  
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.itos = {i:ch for i, ch in enumerate(vocab)}  
        self.stoi = {ch:i for i, ch in enumerate(vocab)}
        self.token_bos = torch.tensor([self.stoi['<|begin_of_text|>']], dtype=torch.int)  
        self.token_eos = torch.tensor([self.stoi['<|end_of_text|>']], dtype=torch.int)  
        self.token_pad = torch.tensor([self.stoi['<|pad_id|>']], dtype=torch.int)
        data = torch.tensor(self.encode(data), dtype=torch.int)
        if split == 'train':
            self.data = data[:int(0.8 * len(data))]
        elif split == 'val':
            self.data = data[int(0.8 * len(data)): int(0.9 * len(data))]
        elif split == 'test':
            self.data = data[int(0.9 * len(data)):]
        elif split == 'full':
            self.data = data
        elif split is None:
            self.data = None
        else:
            raise ValueError(f'{split} is not valid')
        self.seq_len = seq_len
        self.batch_size = batch_size

    # input str，output int list
    def encode(self, s):
        return [self.stoi[ch] for ch in s]  

    # input int list，output str
    def decode(self, l):
        return ''.join(self.itos[i] for i in l)

    def get_batch_data(self):
        # randomly selecting samples from the dataset
        ix = torch.randint(0, len(self.data) - self.seq_len - 3, (self.batch_size,)) 
        x = torch.stack(
            [torch.cat([self.token_bos.clone(), self.data[i:i + self.seq_len - 1]]) for i in ix]
        ).long()
        y = torch.stack(
            [torch.cat([self.data[i + 1:i + self.seq_len], self.token_eos.clone()]) for i in ix]
        ).long()
        return x, y