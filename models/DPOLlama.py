import torch.nn as nn


class DPOLlama(nn.Module):
    def __init__(self, llm):
        super(DPOLlama, self).__init__()
        self.llm = llm
        self.value_head = nn.Linear(llm.params.vocab_size, 1)

    def forward(self, x, targets=None, start_pos=0):
        logits, _ = self.llm(x, targets, start_pos)  # return logits, loss == None
        values = self.value_head(logits).squeeze(-1)
        if not self.training:
            return logits
        return values, logits

    def train(self, mode: bool = True):
        self.training = True
        self.llm.train(mode)
        self.value_head.train(mode)

    def eval(self):
        self.training = False
        self.llm.eval()
        self.value_head.eval()
