import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor.parallel import loss_parallel


class DPOLlama(nn.Module):
    def __init__(self, llm):
        super(DPOLlama, self).__init__()
        self.llm = llm
        self.value_head = nn.Linear(llm.params.vocab_size, 1)

    def forward(self, x, start_pos=0):
        logits, _ = self.llm(x, start_pos)  # return logits, loss == None
        values = self.value_head(logits).squeeze(-1)
        if not self.training:
            return logits
        return values, logits
    
    def dpo_loss(self, values_winner, values_loser, tp:bool, beta=0.1):
        if tp:
            with loss_parallel():
                logits_diff = (values_winner - values_loser) / beta
                loss = -F.logsigmoid(logits_diff).mean()
        else:
            logits_diff = (values_winner - values_loser) / beta
            loss = -F.logsigmoid(logits_diff).mean()
        return loss

    def train(self, mode: bool = True):
        self.training = True
        self.llm.train(mode)
        self.value_head.train(mode)

    def eval(self):
        self.training = False
        self.llm.eval()
        self.value_head.eval()
