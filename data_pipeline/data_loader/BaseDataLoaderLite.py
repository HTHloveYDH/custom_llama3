import os

from data_pipeline.Tokenizer import Tokenizer, ChatFormat


class BaseDataLoaderLite:
    def __init__(self, B, T, process_rank:int, num_processes:int, tokenizer_path:str, data_root:str, \
                 master_process:bool, split:str):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.tokenizer = Tokenizer(tokenizer_path)
        self.chat_format = ChatFormat(self.tokenizer)
        # get filenames
        assert split in {'train', 'val'}
        files = os.listdir(data_root)  # all data files on current node
        split_files = [file for file in files if split in file]
        split_files = sorted(split_files)
        split_files = [os.path.join(data_root, file) for file in split_files]
        self.shards = split_files
        assert len(split_files) > 0, f'no shards found for split {split}'
        if master_process:
            print(f'found {len(split_files)} shards for split {split}')

    def reset(self):
        NotImplementedError(" Can not call 'reset' via base class 'BaseSFTDataLoaderLite'! ")
    
    def next_batch(self):
        NotImplementedError(" Can not call 'next_batch' via base class 'BaseSFTDataLoaderLite'! ")
