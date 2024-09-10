import torch

from config.ModelArgs import ModelArgs
from data_pipeline.demo import DemoDataLoader

 
def generate(model, prompts: str, max_gen_batch_size:int, max_gen_len: int = 500, \
             temperature: float = 0.6, top_p: float = 0.9, device: str = 'cpu'):
    model.eval()
    data_loader = DemoDataLoader('./data/txt', model.params.max_seq_len, max_gen_batch_size, None)
    prompt_tokens = data_loader.token_bos.tolist() + data_loader.encode(prompts)
    assert len(prompt_tokens) <= model.params.max_seq_len, 'should be smaller than max_seq_len'
    total_len = min(len(prompt_tokens) + max_gen_len, model.params.max_seq_len)
    tokens = torch.full(
        (max_gen_batch_size, total_len), fill_value=data_loader.token_pad.item(), dtype=torch.long
    )
    tokens = tokens.to(device)
    tokens[:, :len(prompt_tokens)] = torch.tensor(prompt_tokens, dtype=torch.long).to(device)
    input_text_mask = tokens != data_loader.token_pad.item()
    prev_pos = 0  
    for cur_pos in range(1, total_len):  
        with torch.no_grad():
            logits, _ = model(tokens[:, prev_pos:cur_pos], None, prev_pos)  
        if temperature > 0:        
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)  
            next_token = sample_top_p(probs, top_p)          
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)
        next_token = next_token.reshape(-1)
        next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)  
        tokens[:, cur_pos] = next_token  
        prev_pos = cur_pos
        if (tokens[:, cur_pos] == data_loader.token_pad.item()).all() \
           and \
           (next_token == data_loader.token_eos.item()).all():  
            break
    output_tokens, output_texts = [], []      
    for i, toks in enumerate(tokens.tolist()):  
        if data_loader.token_eos.item() in toks:  
            eos_idx = toks.index(data_loader.token_eos.item())  
            toks = toks[:eos_idx]  

        output_tokens.append(toks)  
        output_texts.append(data_loader.decode(toks))
    # generate sentence
    for i, output_text in enumerate(output_texts):
        output_text = output_text.replace("<|begin_of_text|>", "")  
        print(f'[generation text {i}] ', output_text)
    return output_tokens, output_texts  
 
def sample_top_p(probs, p):  
    probs_sort, prob_idx = torch.sort(probs, dim=-1, descending=True)  
    probs_sum = torch.cumsum(probs_sort, dim=-1)  
    mask = probs_sum-probs_sort > p  
    probs_sort[mask] = 0.0  
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))  
    next_token = torch.multinomial(probs_sort, num_samples=1)  
    next_token = torch.gather(prob_idx, -1, next_token)        
    return next_token