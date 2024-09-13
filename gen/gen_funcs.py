import torch
from torch.nn import functional as F

from utils.get_device_type import get_device_type


def generate(model, tokenizer, chat_format, prompt, device:str, gen_batch_size:int, \
             gen_len:int, dialog:bool, dp_global_rank=0):
    model.eval()
    # preprocess for input prompt: python <class 'list'>
    if dialog:
        assert isinstance(prompt, list)  # example: [{"role": "system", "content": "xxx.",}, {"role": "user", "content": "xxx.",}]
        tokens = chat_format.encode_dialog_prompt(prompt)  # python <class 'list'>
    else:
        assert isinstance(prompt, str)  # "Hello, I am a student."
        tokens = tokenizer.encode(prompt)  # python <class 'list'>
    tokens = torch.tensor(tokens, dtype=torch.long)  # shape: (len(prompt))
    tokens = tokens.unsqueeze(0).repeat(gen_batch_size, 1)  # shape: (gen_batch_size, len(prompt))
    xgen = tokens.to(device)  # current shape: [gen_batch_size, len(prompt))
    xcol = xgen
    # sampling configuration
    device_type = get_device_type(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42 + dp_global_rank)
    while xgen.size(1) < gen_len:
        # forward the model to get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(xcol) # (B, T, vocab_size)
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
    # print the generated text
    for i in range(gen_batch_size):
        tokens = xgen[i, :gen_len].tolist()
        decoded = tokenizer.decode(tokens)
        print(f'[generation text] rank {dp_global_rank} sample {i}: {decoded}')